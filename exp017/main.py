import os
import numpy
import cv2
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

from learning_args import parse_args
from data import generate_images, motion_dict, load_mnist
from models import FullyConvNet, FullyConvResNet, UNet, UNet2, UNet3
from visualize import visualize
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


def validate(args, model, images, m_dict, reverse_m_dict, m_kernel, best_test_acc):
    test_acc = test_supervised(args, model, images, m_dict, reverse_m_dict, m_kernel)
    if test_acc >= best_test_acc:
        logging.info('model save to %s', os.path.join(args.save_dir, 'final.pth'))
        with open(os.path.join(args.save_dir, 'final.pth'), 'w') as handle:
            torch.save(model.state_dict(), handle)
        best_test_acc = test_acc
    logging.info('current best accuracy: %.2f', best_test_acc)
    return best_test_acc


def train_supervised(args, model, images, m_dict, reverse_m_dict, m_kernel):
    m_range = args.motion_range
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_test_acc = 0
    train_loss = []
    for epoch in range(args.train_epoch):
        optimizer.zero_grad()
        im1, im2, im3, gt_motion = generate_images(args, images, m_dict, reverse_m_dict)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, gt_motion = im1.cuda(), im2.cuda(), gt_motion.cuda()
        motion = model(im1, im2)
        motion = motion.transpose(1, 2).transpose(2, 3).contiguous().view(-1, model.n_class)
        gt_motion = gt_motion.contiguous().view(-1)
        loss = F.cross_entropy(motion, gt_motion)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data[0])
        if len(train_loss) > 1000:
            train_loss.pop(0)
        ave_loss = sum(train_loss) / float(len(train_loss))
        logging.info('epoch %d, training loss: %.2f, average training loss: %.2f', epoch, loss.data[0], ave_loss)
        if (epoch+1) % args.test_interval == 0:
            logging.info('epoch %d, testing', epoch)
            best_test_acc = validate(args, model, images, m_dict, reverse_m_dict, m_kernel, best_test_acc)
    return model


def test_supervised(args, model, images, m_dict, reverse_m_dict, m_kernel):
    m_range = args.motion_range
    test_accuracy = []
    for epoch in range(args.test_epoch):
        im1, im2, im3, gt_motion = generate_images(args, images, m_dict, reverse_m_dict)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        im3 = Variable(torch.from_numpy(im3).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, im3, gt_motion = im1.cuda(), im2.cuda(), im3.cuda(), gt_motion.cuda()
        motion = model(im1, im2)
        pred_motion = motion.max(1)[1]
        if args.display:
            m_range = args.motion_range
            pred = construct_image(im2, motion, m_range, m_kernel, padding=m_range)
            visualize(im1, im2, im3, pred, pred_motion, gt_motion, m_range, m_dict, reverse_m_dict)
        # This line assumes disappeared pixels have motion 0, which should be changed in the future.
        pred_motion[pred_motion == model.n_class-1] = m_dict[(0, 0)]
        accuracy = pred_motion.eq(gt_motion).float().sum() * 1.0 / gt_motion.numel()
        test_accuracy.append(accuracy.cpu().data[0])
    test_accuracy = numpy.mean(numpy.asarray(test_accuracy))
    logging.info('average testing accuracy: %.2f', test_accuracy)
    return test_accuracy


def construct_image(im, motion, m_range, m_kernel, padding=0):
    mask = F.softmax(motion)
    im_expand = im.expand_as(mask) * mask
    height = im.size(2) - 2 * m_range + 2 * padding
    width = im.size(3) - 2 * m_range + 2 * padding
    pred = Variable(torch.Tensor(im.size(0), im.size(1), height, width))
    if torch.cuda.is_available():
        pred = pred.cuda()
    for i in range(im.size(0)):
        pred[i, :, :, :] = F.conv2d(im_expand[i, :-1, :, :].unsqueeze(0), m_kernel, None, 1, padding)
    return pred


def train_unsupervised(args, model, images, m_dict, reverse_m_dict, m_kernel):
    m_range = args.motion_range
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_test_acc = 0
    train_loss = []
    for epoch in range(args.train_epoch):
        optimizer.zero_grad()
        im1, im2, im3, gt_motion = generate_images(args, images, m_dict, reverse_m_dict)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        im3 = Variable(torch.from_numpy(im3).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, im3, gt_motion = im1.cuda(), im2.cuda(), im3.cuda(), gt_motion.cuda()
        motion = model(im1, im2)
        pred = construct_image(im2, motion, m_range, m_kernel, padding=0)
        gt = im3[:, :, m_range:-m_range, m_range:-m_range]
        # loss = (pred - gt).pow(2).sum()  # L1 loss is better than L2 loss
        loss = torch.abs(pred - gt).sum()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data[0])
        if len(train_loss) > 1000:
            train_loss.pop(0)
        ave_loss = sum(train_loss) / float(len(train_loss))
        logging.info('epoch %d, training loss: %.2f, average training loss: %.2f', epoch, loss.data[0], ave_loss)
        if (epoch+1) % args.test_interval == 0:
            logging.info('epoch %d, testing', epoch)
            best_test_acc = validate(args, model, images, m_dict, reverse_m_dict, m_kernel, best_test_acc)
    return model


def main():
    args = parse_args()
    logging.info(args)
    m_dict, reverse_m_dict, m_kernel = motion_dict(args.motion_range)
    m_kernel = Variable(torch.from_numpy(m_kernel).float())
    train_images, test_images = load_mnist()
    [_, im_channel, args.image_size, _] = train_images.shape
    model = UNet(args.image_size, im_channel, len(m_dict) + 1)
    if torch.cuda.is_available():
        model = model.cuda()
        m_kernel = m_kernel.cuda()
    if args.train:
        if args.method == 'supervised':
            model = train_supervised(args, model, train_images, m_dict, reverse_m_dict, m_kernel)
        else:
            model = train_unsupervised(args, model, train_images, m_dict, reverse_m_dict, m_kernel)
    if args.test:
        model.load_state_dict(torch.load(args.init_model_path))
        test_supervised(args, model, test_images, m_dict, reverse_m_dict, m_kernel)

if __name__ == '__main__':
    main()
