"""
pytorch                   1.5.1
torchvision               0.6.1
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from cnn import LeNet

import sys
import os
import math

import torch.distributed as dist
from torch.multiprocessing import spawn
from splitdataset import SplitDataset

# DDP = DistributedDataParallel

def init_process(rank, size, fn, pyargs, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1' # localhost/127.0.0.1 or IP of distant machine
    os.environ['MASTER_PORT'] = '29500' # any open ports
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, pyargs)


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM) # all_reduce_multigpu has problem now
        param.grad.data /= size


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(device), target.cuda(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        average_gradients(model) # average gradients from all ranks
        optimizer.step()
        if batch_idx % args.log_interval == 0 and device == 0: # print the results from rank: 0 to reduce massive outputs
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        if args.dry_run:
            break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(device), target.cuda(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if device == 0: # print the results from rank: 0 to reduce massive outputs
        print('\nTest set on {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            device, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def run(rank, size, args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    kwargs = {'batch_size': math.ceil(args.batch_size / size)}
    kwargs.update({'num_workers': 4, 'pin_memory': True, 'shuffle': True})

    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    partition = {str(i): 1 / size for i in range(size)}

    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset_train = SplitDataset(dataset1, partition)
    dataset_train.select(str(rank))
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    dataset_test = SplitDataset(dataset2, partition)
    dataset_test.select(str(rank))

    train_loader = torch.utils.data.DataLoader(dataset_train, **kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **kwargs)

    model = LeNet().cuda(rank)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, train_loader, optimizer, epoch)
        test(model, rank, test_loader)
        scheduler.step()

    if args.save_model and rank == 0:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=5, metavar='S',
                        help='random seed (default: 5)')
    parser.add_argument('--log-interval', type=int, default=15, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    pyargs = parser.parse_args()
    use_cuda = not pyargs.no_cuda and torch.cuda.is_available()

    world_size = 1
    if use_cuda:
        world_size = torch.cuda.device_count()
    else:
        # DDP also supports CPU, but GPU training is more frequent case.
        print('No available GPU instance!')
        sys.exit(0)
    
    _ = datasets.MNIST('./data', train=True, download=True) # download dataset in this example

    # Recommend use `spawn` to initialise DDP on GPUs to avoid issue when multiprocesses(e.g. dataloader) in a forked process
    p = spawn(fn=init_process, args=(world_size, run, pyargs, 'nccl'), nprocs=world_size, join=True)
