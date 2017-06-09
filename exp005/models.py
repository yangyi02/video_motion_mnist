import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class FullyConvNet(nn.Module):
    def __init__(self, im_size, im_channel, n_class):
        super(FullyConvNet, self).__init__()
        num_hidden = 32
        self.conv0 = nn.Conv2d(2*im_channel, num_hidden, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(num_hidden)
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.conv3 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(num_hidden)
        self.conv4 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(num_hidden, num_hidden)
        self.conv5 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(num_hidden)
        self.conv6 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(num_hidden)
        self.conv = nn.Conv2d(num_hidden, n_class, 3, 1, 1)
        self.im_size = im_size
        self.im_channel = im_channel
        self.n_class = n_class

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.bn0(self.conv0(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        motion = self.conv(x)
        return motion


class FullyConvResNet(nn.Module):
    def __init__(self, im_size, im_channel, n_class):
        super(FullyConvNet2, self).__init__()
        num_hidden = 32
        self.conv0 = nn.Conv2d(2*im_channel, num_hidden, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(num_hidden)
        self.conv1_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(num_hidden)
        self.conv1_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(num_hidden)
        self.conv2_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(num_hidden)
        self.conv2_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(num_hidden)
        self.conv3_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(num_hidden)
        self.conv3_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(num_hidden)
        self.conv4_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4_1 = nn.BatchNorm2d(num_hidden)
        self.conv4_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn4_2 = nn.BatchNorm2d(num_hidden)
        self.conv5_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5_1 = nn.BatchNorm2d(num_hidden)
        self.conv5_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn5_2 = nn.BatchNorm2d(num_hidden)
        self.conv6_1 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6_1 = nn.BatchNorm2d(num_hidden)
        self.conv6_2 = nn.Conv2d(num_hidden, num_hidden, 3, 1, 1)
        self.bn6_2 = nn.BatchNorm2d(num_hidden)
        self.conv = nn.Conv2d(num_hidden, n_class, 3, 1, 1)
        self.im_size = im_size
        self.im_channel = im_channel
        self.n_class = n_class

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.bn0(self.conv0(x))
        y = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(y) + x))
        y = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(y) + x))
        y = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(y) + x))
        y = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(y) + x))
        y = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(y) + x))
        y = F.relu(self.bn6_1(self.conv6_1(x)))
        x = F.relu(self.bn6_2(self.conv6_2(y) + x))
        motion = self.conv(x)
        return motion

