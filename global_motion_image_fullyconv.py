import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy
import os
from skimage import io, transform
import matplotlib.pyplot as plt
import h5py


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


def generate_images(m_dict, reverse_m_dict, im_size, im_channel, motion_range, images, batch_size=16):
    # im1 = numpy.random.rand(batch_size, 1, im_size, im_size)
    idx = numpy.random.permutation(images.shape[0])
    im1 = images[idx[0:batch_size], :, :, :]
    with_noise = True
    if with_noise:
        noise = numpy.random.rand(batch_size, 1, im_size, im_size) * 0.0
        im1[im1 == 0] = noise[im1 == 0]
    # im_big = numpy.random.rand(batch_size, 1, im_size+motion_range*2, im_size+motion_range*2)
    im_big = numpy.zeros((batch_size, im_channel, im_size + motion_range * 2, im_size + motion_range * 2))
    im_big[:, :, motion_range:-motion_range, motion_range:-motion_range] = im1
    im2 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    motion_label = numpy.random.randint(0, len(m_dict), size=batch_size)
    gt_motion = numpy.zeros((batch_size, 1, im_size, im_size))
    for i in range(batch_size):
        (motion_x, motion_y) = reverse_m_dict[motion_label[i]]
        im2[i, :, :, :] = im_big[i, :, motion_range+motion_y:motion_range+motion_y+im_size,
              motion_range+motion_x:motion_range+motion_x+im_size]
        gt_motion[i, :, :, :] = motion_label[i]

    im_big = numpy.random.rand(batch_size, im_channel, im_size+motion_range*2, im_size+motion_range*2)
    im_big[:, :, motion_range:-motion_range, motion_range:-motion_range] = im2
    im3 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (motion_x, motion_y) = reverse_m_dict[motion_label[i]]
        im3[i, :, :, :] = im_big[i, :, motion_range + motion_y:motion_range + motion_y + im_size,
                          motion_range + motion_x:motion_range + motion_x + im_size]
    return im1, im2, im3, gt_motion.astype(int)


def load_images(image_size, image_dir='images'):
    script_dir = os.path.dirname(__file__)  # absolute dir the script is in
    image_files = os.listdir(os.path.join(script_dir, image_dir))
    images = []
    for image_file in image_files:
        if image_file.endswith('.jpg'):
            print('loading %s' % image_file)
            image = io.imread(os.path.join(script_dir, image_dir, image_file))
            image = transform.resize(image, (image_size, image_size), mode='constant')
            images.append(image)
    # for i in range(len(images)):
    #     plt.imshow(images[i])
    images = numpy.asarray(images)
    images = images.swapaxes(2, 3).swapaxes(1, 2)
    train_test_split = int(round(len(images) * 0.9))
    train_images = images[0:train_test_split, :, :, :]
    test_images = images[train_test_split:, :, :, :]
    return train_images, test_images


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
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(32, 32)
        self.conv5 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv = nn.Conv2d(32, n_class, 3, 1, 1)
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


class FullyConvNet2(nn.Module):
    def __init__(self, im_size, im_channel, n_class):
        super(FullyConvNet2, self).__init__()
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


def train_supervised(model, motion_range, m_dict, reverse_m_dict, train_images, n_epoch=3000):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(n_epoch):
        optimizer.zero_grad()
        im1, im2, im3, gt_motion = generate_images(m_dict, reverse_m_dict, model.im_size, model.im_channel, motion_range, train_images)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, gt_motion = im1.cuda(), im2.cuda(), gt_motion.cuda()
        motion = model(im1, im2)
        gt_motion = gt_motion[:, :, motion_range:-motion_range, motion_range:-motion_range]
        motion = motion[:, :, motion_range:-motion_range, motion_range:-motion_range]
        motion = motion.transpose(1, 2).transpose(2, 3).contiguous().view(-1, model.n_class)
        gt_motion = gt_motion.contiguous().view(-1)
        loss = F.cross_entropy(motion, gt_motion)
        loss.backward()
        optimizer.step()
        print('epoch %d, training loss: %.2f' % (epoch, loss.data[0]))


def test_supervised(model, motion_range, m_dict, reverse_m_dict, test_images, n_epoch=100):
    val_accuracy = []
    for epoch in range(n_epoch):
        im1, im2, im3, gt_motion = generate_images(m_dict, reverse_m_dict, model.im_size, model.im_channel, motion_range, test_images)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, gt_motion = im1.cuda(), im2.cuda(), gt_motion.cuda()
        motion = model(im1, im2)
        motion = motion.data.max(1)[1]
        accuracy = motion.eq(gt_motion.data).cpu().sum() * 1.0 / motion.numel()
        print('testing accuracy: %.2f' % accuracy)
        val_accuracy.append(accuracy)
        # plt.figure(1)
        # plt.subplot(1,3,1)
        # plt.imshow(im1[0].cpu().data.numpy().squeeze(), cmap='gray')
        # plt.subplot(1,3,2)
        # plt.imshow(im2[0].cpu().data.numpy().squeeze(), cmap='gray')
        # plt.subplot(1,3,3)
        # plt.imshow(motion[0].cpu().numpy().squeeze())
        # plt.show()
    return numpy.mean(numpy.asarray(val_accuracy))


def train_unsupervised(model, motion_range, m_dict, reverse_m_dict, motion_kernel, train_images, n_epoch=2000):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(n_epoch):
        optimizer.zero_grad()
        im1, im2, im3, gt_motion = generate_images(m_dict, reverse_m_dict, model.im_size, model.im_channel, motion_range, train_images)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        im3 = Variable(torch.from_numpy(im3).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, im3, gt_motion = im1.cuda(), im2.cuda(), im3.cuda(), gt_motion.cuda()
        motion = model(im1, im2)
        mask = F.softmax(motion)
        # mask = torch.from_numpy(numpy.zeros((16, 9, 9, 9))).float()
        # for i in range(im1.size()[0]):
        #     motion_label = gt_motion.data[i, 0, 0, 0]
        #     mask[i, motion_label, :, :] = 1
        # im = im1.expand_as(mask) * Variable(mask)
        im = im2.expand_as(mask) * mask
        pred = Variable(torch.Tensor(im2.size()[0], im2.size()[1], im2.size()[2] - 2 * motion_range,
                                     im2.size()[3] - 2 * motion_range))
        if torch.cuda.is_available():
            pred = pred.cuda()
        for i in range(im2.size()[0]):
            pred[i, :, :, :] = F.conv2d(im[i, :, :, :].unsqueeze(0), motion_kernel)
        gt = im3[:, :, motion_range:-motion_range, motion_range:-motion_range]
        loss = (pred - gt).pow(2).sum()
        # cross_entropy_loss = - (mask * torch.log(mask + 0.00001)).sum()
        # loss += 0.01 * cross_entropy_loss
        loss.backward()
        optimizer.step()
        print('epoch %d, training loss: %.2f' % (epoch, loss.data[0]))


def main():
    task = 'mnist'
    im_size, im_channel, motion_range = 28, 3, 1
    m_dict, reverse_m_dict, motion_kernel = motion_dict(motion_range)
    if task == 'mnist':
        train_images, test_images = load_mnist()
        im_size, im_channel = 28, 1
    else:
        train_images, test_images = load_images(im_size)
    model = FullyConvNet2(im_size, im_channel, len(m_dict))
    if torch.cuda.is_available():
        model = model.cuda()
    # train_supervised(model, motion_range, m_dict, reverse_m_dict, train_images)
    # test_accuracy = test_supervised(model, motion_range, m_dict, reverse_m_dict, test_images)
    train_unsupervised(model, motion_range, m_dict, reverse_m_dict, motion_kernel, train_images)
    test_accuracy = test_supervised(model, motion_range, m_dict, reverse_m_dict, test_images)
    print('average testing accuracy: %.2f' % test_accuracy)

if __name__ == '__main__':
    main()
