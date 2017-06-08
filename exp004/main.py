import os
import numpy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt
from learning_args import parse_args
from data import generate_images, motion_dict, load_mnist
from models import Net
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


def train_supervised(args, model, images, m_dict, reverse_m_dict):
    m_range = args.motion_range
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_test_acc = 0
    train_loss = []
    for epoch in range(args.train_epoch):
        optimizer.zero_grad()
        im1, im2, gt_motion = generate_images(args, images, m_dict, reverse_m_dict)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, gt_motion = im1.cuda(), im2.cuda(), gt_motion.cuda()
        motion = model(im1, im2)
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
            best_test_acc = validate(args, model, m_dict, reverse_m_dict, m_kernel, best_test_acc)
    return model


def test_supervised(args, model, images, m_dict, reverse_m_dict):
    test_accuracy = []
    for epoch in range(args.test_epoch):
        im1, im2, gt_motion = generate_images(args, images, m_dict, reverse_m_dict)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, gt_motion = im1.cuda(), im2.cuda(), gt_motion.cuda()
        motion = model(im1, im2)
        motion = motion.data.max(1)[1]
        accuracy = motion.eq(gt_motion.data).cpu().sum() * 1.0 / motion.numel()
        test_accuracy.append(accuracy)
    test_accuracy = numpy.mean(numpy.asarray(test_accuracy))
    logging.info('average testing accuracy: %.2f', test_accuracy)
    return test_accuracy


def train_unsupervised(args, model, images, m_dict, reverse_m_dict):
    m_range = args.motion_range
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.train_epoch):
        optimizer.zero_grad()
        im1, im2, gt_motion = generate_images(args, images, m_dict, reverse_m_dict)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, gt_motion = im1.cuda(), im2.cuda(), gt_motion.cuda()
        motion = model(im1, im2)
        motion = F.softmax(motion)
        motion = motion.view(-1, 1, 2*m_range+1, 2*m_range+1)
        pred = Variable(torch.Tensor(im2.size(0), im2.size(1), im2.size(2) - 2 * m_range,
                                     im2.size(3) - 2 * m_range))
        if torch.cuda.is_available():
            pred = pred.cuda()
        for i in range(im1.size(0)):
            pred[i, :, :, :] = F.conv2d(im1[i, :, :, :].unsqueeze(0), motion[i, :, :, :].unsqueeze(0))
        gt = im2[:, :, m_range:-m_range, m_range:-m_range]
        loss = (pred - gt).pow(2).sum()
        loss.backward()
        optimizer.step()
        logging.info('epoch %d, training loss: %.2f', epoch, loss.data[0])
    return model


def main():
    args = parse_args()
    logging.info(args)
    m_dict, reverse_m_dict = motion_dict(args.motion_range)
    train_images, test_images = load_mnist()
    [_, im_channel, args.image_size, _] = train_images.shape
    model = Net(args.image_size, im_channel, len(m_dict))
    if torch.cuda.is_available():
        model = model.cuda()
    if args.train:
        if args.method == 'supervised':
            model = train_supervised(args, model, train_images, m_dict, reverse_m_dict)
        else:
            model = train_unsupervised(args, model, train_images, m_dict, reverse_m_dict)
    if args.test:
        model.load_state_dict(torch.load(args.init_model_path))
        test_supervised(args, model, test_images, m_dict, reverse_m_dict)

if __name__ == '__main__':
    main()
