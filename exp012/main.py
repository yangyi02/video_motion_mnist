import os
import numpy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from learning_args import parse_args
from data import generate_images, motion_dict, load_mnist
from models import FullyConvNetDecorrelate, FullyConvNetDecorrelate2
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


def train_supervised(args, model, images, m_dict, reverse_m_dict):
    m_range = args.motion_range
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.train_epoch):
        optimizer.zero_grad()
        im1, im2, im3, gt_motion = generate_images(args, images, m_dict, reverse_m_dict)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, gt_motion = im1.cuda(), im2.cuda(), gt_motion.cuda()
        motion_x, motion_y = model(im1, im2)
        mask_x, mask_y = F.softmax(motion_x), F.softmax(motion_y)
        [batch_size, im_channel, height, width] = im2.size()
        mask = Variable(torch.Tensor(batch_size, (2 * m_range + 1) ** 2, height, width))
        if torch.cuda.is_available():
            mask = mask.cuda()
        for i in range(2 * m_range + 1):
            for j in range(2 * m_range + 1):
                mask[:, i * (2 * m_range + 1) + j, :, :] = mask_y[:, i, :, :] * mask_x[:, j, :,
                                                                                     :]
        gt_motion = gt_motion[:, :, m_range:-m_range, m_range:-m_range]
        motion = mask[:, :, m_range:-m_range, m_range:-m_range]
        motion = motion.transpose(1, 2).transpose(2, 3).contiguous().view(-1, model.n_class)
        motion = motion + 1e-5
        motion = torch.log(motion)
        gt_motion = gt_motion.contiguous().view(-1)
        loss = F.cross_entropy(motion, gt_motion)
        loss.backward()
        optimizer.step()
        logging.info('epoch %d, training loss: %.2f', epoch, loss.data[0])
    return model


def test_supervised(args, model, images, m_dict, reverse_m_dict):
    m_range = args.motion_range
    test_accuracy = []
    for epoch in range(args.test_epoch):
        im1, im2, im3, gt_motion = generate_images(args, images, m_dict, reverse_m_dict)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, gt_motion = im1.cuda(), im2.cuda(), gt_motion.cuda()
        motion_x, motion_y = model(im1, im2)
        mask_x, mask_y = F.softmax(motion_x), F.softmax(motion_y)
        [batch_size, im_channel, height, width] = im2.size()
        mask = Variable(torch.Tensor(batch_size, (2 * m_range + 1) ** 2, height, width))
        if torch.cuda.is_available():
            mask = mask.cuda()
        for i in range(2 * m_range + 1):
            for j in range(2 * m_range + 1):
                mask[:, i * (2 * m_range + 1) + j, :, :] = mask_y[:, i, :, :] * mask_x[:, j, :, :]
        motion = mask.data.max(1)[1]
        accuracy = motion.eq(gt_motion.data).cpu().sum() * 1.0 / motion.numel()
        test_accuracy.append(accuracy)
        if args.display:
            visualize(im1, im2, im3, gt_motion, motion)
    test_accuracy = numpy.mean(numpy.asarray(test_accuracy))
    logging.info('average testing accuracy: %.2f', test_accuracy)
    return test_accuracy


def visualize(im1, im2, im3, gt_motion, motion):
    plt.figure(1)
    plt.subplot(1,3,1)
    plt.imshow(im1[0].cpu().data.numpy().squeeze(), cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(im2[0].cpu().data.numpy().squeeze(), cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(motion[0].cpu().numpy().squeeze())
    plt.show()


def train_unsupervised(args, model, images, m_dict, reverse_m_dict, m_kernel):
    m_range = args.motion_range
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.train_epoch):
        optimizer.zero_grad()
        im1, im2, im3, gt_motion = generate_images(args, images, m_dict, reverse_m_dict)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        im3 = Variable(torch.from_numpy(im3).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, im3, gt_motion = im1.cuda(), im2.cuda(), im3.cuda(), gt_motion.cuda()
        motion_x, motion_y = model(im1, im2)
        mask_x, mask_y = F.softmax(motion_x), F.softmax(motion_y)
        [batch_size, im_channel, height, width] = im2.size()
        mask = Variable(torch.Tensor(batch_size, (2 * m_range + 1) ** 2, height, width))
        if torch.cuda.is_available():
            mask = mask.cuda()
        for i in range(2 * m_range + 1):
            for j in range(2 * m_range + 1):
                mask[:, i * (2 * m_range + 1) + j, :, :] = mask_y[:, i, :, :] * mask_x[:, j, :, :]
        im = im2.expand_as(mask) * mask
        pred = Variable(torch.Tensor(im2.size(0), im2.size(1), im2.size(2) - 2 * m_range,
                                     im2.size(3) - 2 * m_range))
        if torch.cuda.is_available():
            pred = pred.cuda()
        for i in range(batch_size):
            pred[i, :, :, :] = F.conv2d(im[i, :, :, :].unsqueeze(0), m_kernel)
        gt = im3[:, :, m_range:-m_range, m_range:-m_range]
        # loss = (pred - gt).pow(2).sum()  # L1 loss is better than L2 loss
        loss = torch.abs(pred - gt).sum()
        loss.backward()
        optimizer.step()
        logging.info('epoch %d, training loss: %.2f', epoch, loss.data[0])
    return model


def main():
    args = parse_args()
    logging.info(args)
    m_dict, reverse_m_dict, m_kernel = motion_dict(args.motion_range)
    m_kernel = Variable(torch.from_numpy(m_kernel).float())
    train_images, test_images = load_mnist()
    [_, im_channel, args.image_size, _] = train_images.shape
    model = FullyConvNetDecorrelate(args.image_size, im_channel, len(m_dict))
    if torch.cuda.is_available():
        model = model.cuda()
        m_kernel = m_kernel.cuda()
    if args.train:
        if args.method == 'supervised':
            model = train_supervised(args, model, train_images, m_dict, reverse_m_dict)
        else:
            model = train_unsupervised(args, model, train_images, m_dict, reverse_m_dict, m_kernel)
        with open(os.path.join(args.save_dir, 'final.pth'), 'w') as handle:
            torch.save(model.state_dict(), handle)
    if args.test:
        model.load_state_dict(torch.load(args.init_model_path))
        test_supervised(args, model, test_images, m_dict, reverse_m_dict)

if __name__ == '__main__':
    main()
