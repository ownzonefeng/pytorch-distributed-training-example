from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        # the structure of convolution net
        self.ConvNet = nn.Sequential(nn.UpsamplingBilinear2d(size=32),
                                     nn.Conv2d(1, 6, 5, padding=0, stride=1),
                                     nn.LeakyReLU(0.2),
                                     nn.MaxPool2d(2, stride=2, padding=0),
                                     nn.Conv2d(6, 16, 5, padding=0, stride=1),
                                     nn.LeakyReLU(0.2),
                                     nn.MaxPool2d(2, stride=2, padding=0),
                                     nn.Conv2d(16, 120, 5, padding=0, stride=1),
                                     nn.LeakyReLU(0.2))

        # the structure of fully connected net
        self.FC = nn.Sequential(nn.Linear(120, 84),
                                nn.LeakyReLU(0.2),
                                nn.Linear(84, 10),
                                nn.Softmax(-1))

    def forward(self, img):
        # the input image passes through network
        output = self.ConvNet(img)
        output = output.view(output.shape[0], -1)
        output = self.FC(output)
        return output
