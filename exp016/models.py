import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


class FullyConvNetDecorrelate(nn.Module):
    def __init__(self, im_size, im_channel, n_class):
        super(FullyConvNetDecorrelate, self).__init__()
        num_hidden = 64
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
        self.conv_x = nn.Conv2d(num_hidden, int(math.sqrt(n_class)), 3, 1, 1)
        self.conv_y = nn.Conv2d(num_hidden, int(math.sqrt(n_class)), 3, 1, 1)
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
        motion_x = self.conv_x(x)
        motion_y = self.conv_y(x)
        return motion_x, motion_y


class FullyConvNetDecorrelate2(nn.Module):
    def __init__(self, im_size, im_channel, n_class):
        super(FullyConvNetDecorrelate2, self).__init__()
        num_hidden = 32
        self.conv0 = nn.Conv2d(im_channel, num_hidden, 3, 1, 1)
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
        self.conv4_1 = nn.Conv2d(2*num_hidden, 2*num_hidden, 3, 1, 1)
        self.bn4_1 = nn.BatchNorm2d(2*num_hidden)
        self.conv4_2 = nn.Conv2d(2*num_hidden, 2*num_hidden, 3, 1, 1)
        self.bn4_2 = nn.BatchNorm2d(2*num_hidden)
        self.conv5_1 = nn.Conv2d(2*num_hidden, 2*num_hidden, 3, 1, 1)
        self.bn5_1 = nn.BatchNorm2d(2*num_hidden)
        self.conv5_2 = nn.Conv2d(2*num_hidden, 2*num_hidden, 3, 1, 1)
        self.bn5_2 = nn.BatchNorm2d(2*num_hidden)
        self.conv6_1 = nn.Conv2d(2*num_hidden, 2*num_hidden, 3, 1, 1)
        self.bn6_1 = nn.BatchNorm2d(2*num_hidden)
        self.conv6_2 = nn.Conv2d(2*num_hidden, 2*num_hidden, 3, 1, 1)
        self.bn6_2 = nn.BatchNorm2d(2*num_hidden)
        self.conv_x = nn.Conv2d(2*num_hidden, int(math.sqrt(n_class)), 3, 1, 1)
        self.conv_y = nn.Conv2d(2*num_hidden, int(math.sqrt(n_class)), 3, 1, 1)
        self.im_size = im_size
        self.im_channel = im_channel
        self.n_class = n_class

    def forward(self, x1, x2):
        x1 = self.bn0(self.conv0(x1))
        y1 = F.relu(self.bn1_1(self.conv1_1(x1)))
        x1 = F.relu(self.bn1_2(self.conv1_2(y1) + x1))
        y1 = F.relu(self.bn2_1(self.conv2_1(x1)))
        x1 = F.relu(self.bn2_2(self.conv2_2(y1) + x1))
        y1 = F.relu(self.bn3_1(self.conv3_1(x1)))
        x1 = F.relu(self.bn3_2(self.conv3_2(y1) + x1))
        x2 = self.bn0(self.conv0(x2))
        y2 = F.relu(self.bn1_1(self.conv1_1(x2)))
        x2 = F.relu(self.bn1_2(self.conv1_2(y2) + x2))
        y2 = F.relu(self.bn2_1(self.conv2_1(x2)))
        x2 = F.relu(self.bn2_2(self.conv2_2(y2) + x2))
        y2 = F.relu(self.bn3_1(self.conv3_1(x2)))
        x2 = F.relu(self.bn3_2(self.conv3_2(y2) + x2))
        x = torch.cat((x1, x2), 1)
        y = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(y) + x))
        y = F.relu(self.bn5_1(self.conv5_1(x)))
        x = F.relu(self.bn5_2(self.conv5_2(y) + x))
        y = F.relu(self.bn6_1(self.conv6_1(x)))
        x = F.relu(self.bn6_2(self.conv6_2(y) + x))
        motion_x = self.conv_x(x)
        motion_y = self.conv_y(x)
        return motion_x, motion_y
