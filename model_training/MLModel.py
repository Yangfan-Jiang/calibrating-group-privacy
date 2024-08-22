# Machine learning models
import torch
from torch import nn
from kymatio.torch import Scattering2D



SHAPES = {
    "cifar10": (32, 32, 3),
    "cifar10_500K": (32, 32, 3),
    "fmnist": (28, 28, 1),
    "mnist": (28, 28, 1)
}


def get_scatter_transform(dataset="mnist"):
    shape = SHAPES[dataset]
    scattering = Scattering2D(J=2, shape=shape[:2])
    K = 81 * shape[2]
    (h, w) = shape[:2]
    return scattering, K, (h//4, w//4)


class ScatterLinear(nn.Module):
    """
    ScatterNet model used in the following paper
    - Tramer, Florian, and Dan Boneh. Differentially Private Learning Needs Better Features (or Much More Data). In ICLR 2021. 
    See https://github.com/ftramer/Handcrafted-DP/blob/main/models.py
    """
    def __init__(self, in_channels, hw_dims, input_norm=None, classes=10, clip_norm=None, **kwargs):
        super(ScatterLinear, self).__init__()
        self.K = in_channels
        self.h = hw_dims[0]
        self.w = hw_dims[1]
        self.fc = None
        self.norm = None
        self.clip = None
        self.build(input_norm, classes=classes, clip_norm=clip_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None, bn_stats=None, clip_norm=None, classes=10):
        self.fc = nn.Linear(self.K * self.h * self.w, classes)

        if input_norm is None:
            self.norm = nn.Identity()
        elif input_norm == "GroupNorm":
            self.norm = nn.GroupNorm(num_groups, self.K, affine=False)
        else:
            self.norm = lambda x: standardize(x, bn_stats)

        if clip_norm is None:
            self.clip = nn.Identity()
        else:
            self.clip = ClipLayer(clip_norm)

    def forward(self, x):
        x = self.norm(x.view(-1, self.K, self.h, self.w))
        x = self.clip(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    
class LogisticRegression(nn.Module):
    """Logistic regression"""
    def __init__(self, num_feature, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_feature, output_size)

    def forward(self, x):
        return self.linear(x)

    
class CIFAR10_CNN(nn.Module):
    def __init__(self, in_channels=3, input_norm=None, **kwargs):
        super(CIFAR10_CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):

        if self.in_channels == 3:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32, 'M', 64, 'M']
            else:
                cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 'M']

            self.norm = nn.Identity()
        else:
            if size == "small":
                cfg = [16, 16, 'M', 32, 32]
            else:
                cfg = [64, 'M', 64]
            if input_norm is None:
                self.norm = nn.Identity()
            elif input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            else:
                self.norm = lambda x: standardize(x, bn_stats)

        layers = []
        act = nn.Tanh

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(c, v, kernel_size=3, stride=1, padding=1)

                layers += [conv2d, act()]
                c = v

        self.features = nn.Sequential(*layers)

        if self.in_channels == 3:
            hidden = 128
            self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden), act(), nn.Linear(hidden, 10))
        else:
            self.classifier = nn.Linear(c * 4 * 4, 10)

    def forward(self, x):
        if self.in_channels != 3:
            x = self.norm(x.view(-1, self.in_channels, 8, 8))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        print("#. features=", x.shape)
        x = self.classifier(x)
        return x


class MNIST_CNN(nn.Module):
    def __init__(self, in_channels=1, input_norm=None, **kwargs):
        super(MNIST_CNN, self).__init__()
        self.in_channels = in_channels
        self.features = None
        self.classifier = None
        self.norm = None

        self.build(input_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None,
              bn_stats=None, size=None):
        if self.in_channels == 1:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 8, 2, 2), 'M', (ch2, 4, 2, 0), 'M']
            self.norm = nn.Identity()
        else:
            ch1, ch2 = (16, 32) if size is None else (32, 64)
            cfg = [(ch1, 3, 2, 1), (ch2, 3, 1, 1)]
            if input_norm == "GroupNorm":
                self.norm = nn.GroupNorm(num_groups, self.in_channels, affine=False)
            elif input_norm == "BN":
                self.norm = lambda x: standardize(x, bn_stats)
            else:
                self.norm = nn.Identity()

        layers = []

        c = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=1)]
            else:
                filters, k_size, stride, pad = v
                conv2d = nn.Conv2d(c, filters, kernel_size=k_size, stride=stride, padding=pad)

                layers += [conv2d, nn.Tanh()]
                c = filters

        self.features = nn.Sequential(*layers)

        hidden = 32
        self.classifier = nn.Sequential(nn.Linear(c * 4 * 4, hidden),
                                        nn.Tanh(),
                                        nn.Linear(hidden, 10))

    def forward(self, x):
        if self.in_channels != 1:
            x = self.norm(x.view(-1, self.in_channels, 7, 7))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        #print("#. features=", x.shape)
        x = self.classifier(x)
        return x
    
    
def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma * s, data_shape).to(device)



