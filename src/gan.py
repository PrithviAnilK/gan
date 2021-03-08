import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 128 * 12 * 12)
        self.bn = nn.BatchNorm1d(128 * 12 * 12)

        self.cvt1 = nn.ConvTranspose2d(128, 64, kernel_size = 5, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.cvt2 = nn.ConvTranspose2d(64, 32, kernel_size = 7, bias = False)
        self.bn2 = nn.BatchNorm2d(32)
        self.cvt3 = nn.ConvTranspose2d(32, 1, kernel_size = 7)

    def forward(self, x):
        x = F.relu(self.bn(self.fc(x)))
        x = x.view(-1, 128, 12, 12)
        x = F.relu(self.bn1(self.cvt1(x)))
        x = F.relu(self.bn2(self.cvt2(x)))
        x = torch.tanh(self.cvt3(x))
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cv1 = nn.Conv2d(1, 32, kernel_size = 5, stride = 2, padding = 2, bias = False)
        self.bn1 = nn.BatchNorm2d(32)
        self.cv2 = nn.Conv2d(32, 64, kernel_size = 5, stride = 2, padding = 2, bias = False)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 1)

    def flatten(self, x):
        bs = x.size()[0]
        x = x.view(-1, int(x.numel() / bs))
        return x

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.cv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.cv2(x)), 0.2)
        x = self.flatten(x)
        x = self.fc(x)
        return x 