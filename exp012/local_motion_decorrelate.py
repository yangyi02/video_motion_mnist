import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy
import os
import matplotlib.pyplot as plt
import h5py
import math


def motion_dict(motion_range):
    m_dict, reverse_m_dict = {}, {}
    x = numpy.linspace(-motion_range, motion_range, 2*motion_range+1)
    y = numpy.linspace(-motion_range, motion_range, 2*motion_range+1)
    m_x, m_y = numpy.meshgrid(x, y)
    m_x, m_y, = m_x.reshape(-1).astype(int), m_y.reshape(-1).astype(int)
    motion_kernel = Variable(torch.zeros((1, len(m_x), 2*motion_range+1, 2*motion_range+1)))
    if torch.cuda.is_available():
        motion_kernel = motion_kernel.cuda()
    for i in range(len(m_x)):
        m_dict[(m_x[i], m_y[i])] = i
        reverse_m_dict[i] = (m_x[i], m_y[i])
        motion_kernel[:, i, m_y[i]+motion_range, m_x[i]+motion_range] = 1
    return m_dict, reverse_m_dict, motion_kernel


def generate_images1(m_dict, reverse_m_dict, im_size, im_channel, motion_range, images, batch_size=32):
    show = False
    noise_magnitude = 0.5
    # im1 = numpy.random.rand(batch_size, im_channel, im_size, im_size)
    idx = numpy.random.permutation(images.shape[0])
    im1 = images[idx[0:batch_size], :, :, :]
    bg1 = numpy.random.rand(batch_size, im_channel, im_size, im_size) * noise_magnitude
    if show:
        plt.figure(1)
        plt.subplot(1,2,1)
        plt.imshow(im1[0, :, :, :].squeeze(), cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(bg1[0, :, :, :].squeeze())
        # plt.show()

    # im_big = numpy.random.rand(batch_size, 1, im_size+motion_range*2, im_size+motion_range*2)
    im_big = numpy.zeros((batch_size, im_channel, im_size+motion_range*2, im_size+motion_range*2))
    im_big[:, :, motion_range:-motion_range, motion_range:-motion_range] = im1
    im2 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    motion_label = numpy.random.randint(0, len(m_dict), size=batch_size)
    gt_motion = numpy.zeros((batch_size, 1, im_size, im_size))
    for i in range(batch_size):
        (motion_x, motion_y) = reverse_m_dict[motion_label[i]]
        im2[i, :, :, :] = im_big[i, :, motion_range+motion_y:motion_range+motion_y+im_size,
              motion_range+motion_x:motion_range+motion_x+im_size]
        gt_motion[i, :, :, :] = motion_label[i]
    bg_big = numpy.random.rand(batch_size, im_channel, im_size+motion_range*2, im_size+motion_range*2) * noise_magnitude
    bg_big[:, :, motion_range:-motion_range, motion_range:-motion_range] = bg1
    bg2 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    bg_motion_label = numpy.random.randint(0, len(m_dict), size=batch_size)
    bg_gt_motion = numpy.zeros((batch_size, 1, im_size, im_size))
    for i in range(batch_size):
        (bg_motion_x, bg_motion_y) = reverse_m_dict[bg_motion_label[i]]
        bg2[i, :, :, :] = bg_big[i, :, motion_range+bg_motion_y:motion_range+bg_motion_y+im_size,
              motion_range+bg_motion_x:motion_range+bg_motion_x+im_size]
        bg_gt_motion[i, :, :, :] = bg_motion_label[i]
    if show:
        plt.figure(2)
        plt.subplot(1,2,1)
        plt.imshow(im2[0, :, :, :].squeeze(), cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(bg2[0, :, :, :].squeeze())
        # plt.show()

    im_big = numpy.zeros((batch_size, im_channel, im_size+motion_range*2, im_size+motion_range*2))
    im_big[:, :, motion_range:-motion_range, motion_range:-motion_range] = im2
    im3 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (motion_x, motion_y) = reverse_m_dict[motion_label[i]]
        im3[i, :, :, :] = im_big[i, :, motion_range + motion_y:motion_range+motion_y+im_size,
                          motion_range+motion_x:motion_range+motion_x+im_size]
    bg_big = numpy.random.rand(batch_size, im_channel, im_size+motion_range*2, im_size+motion_range*2) * noise_magnitude
    bg_big[:, :, motion_range:-motion_range, motion_range:-motion_range] = bg2
    bg3 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (bg_motion_x, bg_motion_y) = reverse_m_dict[bg_motion_label[i]]
        bg3[i, :, :, :] = bg_big[i, :, motion_range+bg_motion_y:motion_range+bg_motion_y+im_size,
              motion_range+bg_motion_x:motion_range+bg_motion_x+im_size]
    if show:
        plt.figure(3)
        plt.subplot(1,2,1)
        plt.imshow(im3[0, :, :, :].squeeze(), cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(bg3[0, :, :, :].squeeze())
        # plt.show()

    gt_motion[im2 == 0] = bg_gt_motion[im2 == 0]
    im1[im1 == 0] = bg1[im1 == 0]
    im2[im2 == 0] = bg2[im2 == 0]
    im3[im3 == 0] = bg3[im3 == 0]
    if show:
        plt.figure(4)
        plt.subplot(2,2,1)
        plt.imshow(im1[0, :, :, :].squeeze())
        plt.subplot(2,2,2)
        plt.imshow(im2[0, :, :, :].squeeze())
        plt.subplot(2,2,3)
        plt.imshow(im3[0, :, :, :].squeeze())
        plt.subplot(2,2,4)
        plt.imshow(gt_motion[0, :, :, :].squeeze())
        plt.show()

    return im1, im2, im3, gt_motion.astype(int)


def generate_images(m_dict, reverse_m_dict, im_size, im_channel, motion_range, images, batch_size=32):
    show = False
    noise_magnitude = 0.5
    # im1 = numpy.random.rand(batch_size, im_channel, im_size, im_size)
    idx = numpy.random.permutation(images.shape[0])
    im1_1 = images[idx[0:batch_size], :, :, :]
    bg1_1 = numpy.random.rand(batch_size, im_channel, im_size, im_size) * noise_magnitude
    idx = numpy.random.permutation(images.shape[0])
    im1_2 = images[idx[0:batch_size], :, :, :]
    bg1_2 = numpy.random.rand(batch_size, im_channel, im_size, im_size) * noise_magnitude
    if show:
        plt.figure(1)
        plt.subplot(1,2,1)
        plt.imshow(numpy.concatenate((im1_1[0, :, :, :].squeeze(), im1_2[0, :, :, :].squeeze()), 1), cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(numpy.concatenate((bg1_1[0, :, :, :].squeeze(), bg1_2[0, :, :, :].squeeze()), 1))
        # plt.show()

    # im_big = numpy.random.rand(batch_size, 1, im_size+motion_range*2, im_size+motion_range*2)
    im_big = numpy.zeros((batch_size, im_channel, im_size+motion_range*2, im_size+motion_range*2))
    im_big[:, :, motion_range:-motion_range, motion_range:-motion_range] = im1_1
    im2_1 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    motion_label1 = numpy.random.randint(0, len(m_dict), size=batch_size)
    gt_motion1 = numpy.zeros((batch_size, 1, im_size, im_size))
    for i in range(batch_size):
        (motion_x, motion_y) = reverse_m_dict[motion_label1[i]]
        im2_1[i, :, :, :] = im_big[i, :, motion_range+motion_y:motion_range+motion_y+im_size,
              motion_range+motion_x:motion_range+motion_x+im_size]
        gt_motion1[i, :, :, :] = motion_label1[i]
    im_big = numpy.zeros(
        (batch_size, im_channel, im_size + motion_range * 2, im_size + motion_range * 2))
    im_big[:, :, motion_range:-motion_range, motion_range:-motion_range] = im1_2
    im2_2 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    motion_label2 = numpy.random.randint(0, len(m_dict), size=batch_size)
    gt_motion2 = numpy.zeros((batch_size, 1, im_size, im_size))
    for i in range(batch_size):
        (motion_x, motion_y) = reverse_m_dict[motion_label2[i]]
        im2_2[i, :, :, :] = im_big[i, :, motion_range + motion_y:motion_range + motion_y + im_size,
                            motion_range + motion_x:motion_range + motion_x + im_size]
        gt_motion2[i, :, :, :] = motion_label2[i]
    bg_motion_label = numpy.random.randint(0, len(m_dict), size=batch_size)
    bg_gt_motion1 = numpy.zeros((batch_size, 1, im_size, im_size))
    bg_big = numpy.random.rand(batch_size, im_channel, im_size+motion_range*2, im_size+motion_range*2) * noise_magnitude
    bg_big[:, :, motion_range:-motion_range, motion_range:-motion_range] = bg1_1
    bg2_1 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (bg_motion_x, bg_motion_y) = reverse_m_dict[bg_motion_label[i]]
        bg2_1[i, :, :, :] = bg_big[i, :, motion_range+bg_motion_y:motion_range+bg_motion_y+im_size,
              motion_range+bg_motion_x:motion_range+bg_motion_x+im_size]
        bg_gt_motion1[i, :, :, :] = bg_motion_label[i]
    bg_gt_motion2 = numpy.zeros((batch_size, 1, im_size, im_size))
    bg_big = numpy.random.rand(batch_size, im_channel, im_size + motion_range * 2,
                                im_size + motion_range * 2) * noise_magnitude
    bg_big[:, :, motion_range:-motion_range, motion_range:-motion_range] = bg1_2
    bg2_2 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (bg_motion_x, bg_motion_y) = reverse_m_dict[bg_motion_label[i]]
        bg2_2[i, :, :, :] = bg_big[i, :,
                            motion_range + bg_motion_y:motion_range + bg_motion_y + im_size,
                            motion_range + bg_motion_x:motion_range + bg_motion_x + im_size]
        bg_gt_motion2[i, :, :, :] = bg_motion_label[i]
    if show:
        plt.figure(2)
        plt.subplot(1,2,1)
        plt.imshow(numpy.concatenate((im2_1[0, :, :, :].squeeze(), im2_2[0, :, :, :].squeeze()), 1), cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(numpy.concatenate((bg2_1[0, :, :, :].squeeze(), bg2_2[0, :, :, :].squeeze()), 1))
        # plt.show()

    im_big = numpy.zeros((batch_size, im_channel, im_size+motion_range*2, im_size+motion_range*2))
    im_big[:, :, motion_range:-motion_range, motion_range:-motion_range] = im2_1
    im3_1 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (motion_x, motion_y) = reverse_m_dict[motion_label1[i]]
        im3_1[i, :, :, :] = im_big[i, :, motion_range + motion_y:motion_range+motion_y+im_size,
                          motion_range+motion_x:motion_range+motion_x+im_size]
    im_big = numpy.zeros((batch_size, im_channel, im_size+motion_range*2, im_size+motion_range*2))
    im_big[:, :, motion_range:-motion_range, motion_range:-motion_range] = im2_2
    im3_2 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (motion_x, motion_y) = reverse_m_dict[motion_label1[i]]
        im3_2[i, :, :, :] = im_big[i, :, motion_range + motion_y:motion_range+motion_y+im_size,
                          motion_range+motion_x:motion_range+motion_x+im_size]
    bg_big = numpy.random.rand(batch_size, im_channel, im_size+motion_range*2, im_size+motion_range*2) * noise_magnitude
    bg_big[:, :, motion_range:-motion_range, motion_range:-motion_range] = bg2_1
    bg3_1 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (bg_motion_x, bg_motion_y) = reverse_m_dict[bg_motion_label[i]]
        bg3_1[i, :, :, :] = bg_big[i, :, motion_range+bg_motion_y:motion_range+bg_motion_y+im_size,
              motion_range+bg_motion_x:motion_range+bg_motion_x+im_size]
    bg_big = numpy.random.rand(batch_size, im_channel, im_size + motion_range * 2,
                               im_size + motion_range * 2) * noise_magnitude
    bg_big[:, :, motion_range:-motion_range, motion_range:-motion_range] = bg2_2
    bg3_2 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (bg_motion_x, bg_motion_y) = reverse_m_dict[bg_motion_label[i]]
        bg3_2[i, :, :, :] = bg_big[i, :,
                            motion_range + bg_motion_y:motion_range + bg_motion_y + im_size,
                            motion_range + bg_motion_x:motion_range + bg_motion_x + im_size]
    if show:
        plt.figure(3)
        plt.subplot(1,2,1)
        plt.imshow(numpy.concatenate((im3_1[0, :, :, :].squeeze(), im3_2[0, :, :, :].squeeze()), 1), cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(numpy.concatenate((bg3_1[0, :, :, :].squeeze(), bg3_2[0, :, :, :].squeeze()), 1))
        # plt.show()

    gt_motion1[im2_1 == 0] = bg_gt_motion1[im2_1 == 0]
    gt_motion2[im2_2 == 0] = bg_gt_motion2[im2_2 == 0]
    im1_1[im1_1 == 0] = bg1_1[im1_1 == 0]
    im1_2[im1_2 == 0] = bg1_2[im1_2 == 0]
    im2_1[im2_1 == 0] = bg2_1[im2_1 == 0]
    im2_2[im2_2 == 0] = bg2_2[im2_2 == 0]
    im3_1[im3_1 == 0] = bg3_1[im3_1 == 0]
    im3_2[im3_2 == 0] = bg3_2[im3_2 == 0]
    im1 = numpy.concatenate((im1_1, im1_2), 3)
    im2 = numpy.concatenate((im2_1, im2_2), 3)
    im3 = numpy.concatenate((im3_1, im3_2), 3)
    gt_motion = numpy.concatenate((gt_motion1, gt_motion2), 3)
    if show:
        plt.figure(4)
        plt.subplot(2,2,1)
        plt.imshow(im1[0, :, :, :].squeeze())
        plt.subplot(2,2,2)
        plt.imshow(im2[0, :, :, :].squeeze())
        plt.subplot(2,2,3)
        plt.imshow(im3[0, :, :, :].squeeze())
        plt.subplot(2,2,4)
        plt.imshow(gt_motion[0, :, :, :].squeeze())
        plt.show()

    return im1, im2, im3, gt_motion.astype(int)


def load_mnist(file_name='mnist.h5'):
    script_dir = os.path.dirname(__file__)  # absolute dir the script is in
    f = h5py.File(os.path.join(script_dir, file_name))
    train_images = f['train'].value.reshape(-1, 28, 28)
    train_images = numpy.expand_dims(train_images, 1)
    test_images = f['test'].value.reshape(-1, 28, 28)
    test_images = numpy.expand_dims(test_images, 1)
    return train_images, test_images


class FullyConvNet(nn.Module):
    def __init__(self, im_size, im_channel, n_class):
        super(FullyConvNet, self).__init__()
        self.conv0 = nn.Conv2d(2*im_channel, 32, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(32)
        self.conv1_1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.conv2_1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(32)
        self.conv3_1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn3_1 = nn.BatchNorm2d(32)
        self.conv3_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(32)
        self.conv4_1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn4_1 = nn.BatchNorm2d(32, 32)
        self.conv4_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn4_2 = nn.BatchNorm2d(32)
        self.conv5_1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn5_1 = nn.BatchNorm2d(32)
        self.conv5_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn5_2 = nn.BatchNorm2d(32)
        self.conv6_1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn6_1 = nn.BatchNorm2d(32)
        self.conv6_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn6_2 = nn.BatchNorm2d(32)
        self.conv = nn.Conv2d(32, n_class, 3, 1, 1)
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


class FullyConvNet2Decorrelate(nn.Module):
    def __init__(self, im_size, im_channel, n_class):
        super(FullyConvNet2Decorrelate, self).__init__()
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


def train_supervised(model, motion_range, m_dict, reverse_m_dict, train_images, n_epoch=2000):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epoch):
        optimizer.zero_grad()
        im1, im2, im3, gt_motion = generate_images(m_dict, reverse_m_dict, model.im_size, model.im_channel, motion_range, train_images)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, gt_motion = im1.cuda(), im2.cuda(), gt_motion.cuda()
        motion_x, motion_y = model(im1, im2)
        mask_x, mask_y = F.softmax(motion_x), F.softmax(motion_y)
        [batch_size, im_channel, height, width] = im2.size()
        mask = Variable(torch.Tensor(batch_size, (2 * motion_range + 1) ** 2, height, width))
        if torch.cuda.is_available():
            mask = mask.cuda()
        for i in range(2 * motion_range + 1):
            for j in range(2 * motion_range + 1):
                mask[:, i * (2 * motion_range + 1) + j, :, :] = mask_y[:, i, :, :] * mask_x[:, j, :, :]
        gt_motion = gt_motion[:, :, motion_range:-motion_range, motion_range:-motion_range]
        motion = mask[:, :, motion_range:-motion_range, motion_range:-motion_range]
        motion = motion.transpose(1, 2).transpose(2, 3).contiguous().view(-1, model.n_class)
        motion = motion + 1e-5
        motion = torch.log(motion)
        gt_motion = gt_motion.contiguous().view(-1)
        # loss = F.cross_entropy(motion, gt_motion)
        loss = F.nll_loss(motion, gt_motion)
        loss.backward()
        optimizer.step()
        print('epoch %d, training loss: %.2f' % (epoch, loss.data[0]))


def test_supervised(model, motion_range, m_dict, reverse_m_dict, test_images, n_epoch=100):
    show = False
    val_accuracy = []
    for epoch in range(n_epoch):
        im1, im2, im3, gt_motion = generate_images(m_dict, reverse_m_dict, model.im_size, model.im_channel, motion_range, test_images)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, gt_motion = im1.cuda(), im2.cuda(), gt_motion.cuda()
        motion_x, motion_y = model(im1, im2)
        mask_x, mask_y = F.softmax(motion_x), F.softmax(motion_y)
        [batch_size, im_channel, height, width] = im2.size()
        mask = Variable(torch.Tensor(batch_size, (2 * motion_range + 1) ** 2, height, width))
        if torch.cuda.is_available():
            mask = mask.cuda()
        for i in range(2 * motion_range + 1):
            for j in range(2 * motion_range + 1):
                mask[:, i * (2 * motion_range + 1) + j, :, :] = mask_y[:, i, :, :] * mask_x[:, j, :,:]
        motion = mask.data.max(1)[1]
        accuracy = motion.eq(gt_motion.data).cpu().sum() * 1.0 / motion.numel()
        print('testing accuracy: %.2f' % accuracy)
        val_accuracy.append(accuracy)
        if show:
            plt.figure(1)
            plt.subplot(1,3,1)
            plt.imshow(im1[0].cpu().data.numpy().squeeze(), cmap='gray')
            plt.subplot(1,3,2)
            plt.imshow(im2[0].cpu().data.numpy().squeeze(), cmap='gray')
            plt.subplot(1,3,3)
            plt.imshow(motion[0].cpu().numpy().squeeze())
            plt.show()
    return numpy.mean(numpy.asarray(val_accuracy))


def train_unsupervised(model, motion_range, m_dict, reverse_m_dict, motion_kernel, train_images, n_epoch=1000):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epoch):
        optimizer.zero_grad()
        im1, im2, im3, gt_motion = generate_images(m_dict, reverse_m_dict, model.im_size, model.im_channel, motion_range, train_images)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        im3 = Variable(torch.from_numpy(im3).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, im3, gt_motion = im1.cuda(), im2.cuda(), im3.cuda(), gt_motion.cuda()
        motion_x, motion_y = model(im1, im2)
        mask_x, mask_y = F.softmax(motion_x), F.softmax(motion_y)
        # mask = torch.from_numpy(numpy.zeros((16, 9, 9, 9))).float()
        # for i in range(im1.size()[0]):
        #     motion_label = gt_motion.data[i, 0, 0, 0]
        #     mask[i, motion_label, :, :] = 1
        # im = im1.expand_as(mask) * Variable(mask)
        [batch_size, im_channel, height, width] = im2.size()
        # mask_x = mask_x.unsqueeze(1).expand((batch_size, 2*motion_range+1, 2*motion_range+1, height, width))
        # mask_y = mask_y.unsqueeze(2).expand((batch_size, 2*motion_range+1, 2*motion_range+1, height, width))
        # mask = mask_x * mask_y
        # mask = mask.view(batch_size, -1, height, width)
        mask = Variable(torch.Tensor(batch_size, (2*motion_range+1)**2, height, width))
        if torch.cuda.is_available():
            mask = mask.cuda()
        for i in range(2*motion_range+1):
            for j in range(2*motion_range+1):
                mask[:, i*(2*motion_range+1)+j, :, :] = mask_y[:, i, :, :] * mask_x[:, j, :, :]
        im = im2.expand_as(mask) * mask
        # pred = Variable(torch.Tensor(im2.size()[0], im2.size()[1], im2.size()[2] - 2 * motion_range,
        #                              im2.size()[3] - 2 * motion_range))
        pred = Variable(torch.Tensor(batch_size, im_channel, height-2*motion_range, width-2*motion_range))
        if torch.cuda.is_available():
            pred = pred.cuda()
        for i in range(batch_size):
            pred[i, :, :, :] = F.conv2d(im[i, :, :, :].unsqueeze(0), motion_kernel)
        gt = im3[:, :, motion_range:-motion_range, motion_range:-motion_range]
        # loss = (pred - gt).pow(2).sum()  # L1 loss is better than L2 loss
        loss = torch.abs(pred - gt).sum()
        # cross_entropy_loss = - (mask * torch.log(mask + 0.00001)).sum()
        # loss += 0.01 * cross_entropy_loss
        loss.backward()
        optimizer.step()
        print('epoch %d, training loss: %.2f' % (epoch, loss.data[0]))


def main():
    task = 'mnist'
    im_size, im_channel, motion_range = 28, 1, 2
    m_dict, reverse_m_dict, motion_kernel = motion_dict(motion_range)
    train_images, test_images = load_mnist()
    model = FullyConvNetDecorrelate(im_size, im_channel, len(m_dict))
    if torch.cuda.is_available():
        model = model.cuda()
    # train_supervised(model, motion_range, m_dict, reverse_m_dict, train_images)
    # test_accuracy = test_supervised(model, motion_range, m_dict, reverse_m_dict, test_images)
    train_unsupervised(model, motion_range, m_dict, reverse_m_dict, motion_kernel, train_images)
    test_accuracy = test_supervised(model, motion_range, m_dict, reverse_m_dict, test_images)
    print('average testing accuracy: %.2f' % test_accuracy)

if __name__ == '__main__':
    main()
