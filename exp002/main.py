import os
import numpy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from learning_args import parse_args
from data import generate_images, motion_dict
from models import FullyConvNet, FullyConvNet2
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


def train_supervised(args, model, m_dict, reverse_m_dict):
    m_range = args.motion_range
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.train_epoch):
        optimizer.zero_grad()
        im1, im2, gt_motion = generate_images(args, m_dict, reverse_m_dict)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, gt_motion = im1.cuda(), im2.cuda(), gt_motion.cuda()
        motion = model(im1, im2)
        gt_motion = gt_motion[:, :, m_range:-m_range, m_range:-m_range]
        motion = motion[:, :, m_range:-m_range, m_range:-m_range]
        motion = motion.transpose(1, 2).transpose(2, 3).contiguous().view(-1, model.n_class)
        gt_motion = gt_motion.contiguous().view(-1)
        loss = F.cross_entropy(motion, gt_motion)
        loss.backward()
        optimizer.step()
        logging.info('epoch %d, training loss: %.2f', epoch, loss.data[0])
    return model


def test_supervised(args, model, m_dict, reverse_m_dict):
    test_accuracy = []
    for epoch in range(args.test_epoch):
        im1, im2, gt_motion = generate_images(args, m_dict, reverse_m_dict)
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


def train_unsupervised(args, model, m_dict, reverse_m_dict, m_kernel):
    m_range = args.motion_range
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.train_epoch):
        optimizer.zero_grad()
        im1, im2, gt_motion = generate_images(args, m_dict, reverse_m_dict)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, gt_motion = im1.cuda(), im2.cuda(), gt_motion.cuda()
        motion = model(im1, im2)
        mask = F.softmax(motion)
        # mask = torch.from_numpy(numpy.zeros((16, 9, 9, 9))).float()
        # for i in range(im1.size()[0]):
        #     motion_label = gt_motion.data[i, 0, 0, 0]
        #     mask[i, motion_label, :, :] = 1
        # im = im1.expand_as(mask) * Variable(mask)
        im = im1.expand_as(mask) * mask
        pred = Variable(torch.Tensor(im2.size(0), im2.size(1), im2.size(2) - 2 * m_range,
                                     im2.size(3) - 2 * m_range))
        if torch.cuda.is_available():
            pred = pred.cuda()
        for i in range(im1.size(0)):
            pred[i, :, :, :] = F.conv2d(im[i, :, :, :].unsqueeze(0), m_kernel)
        gt = im2[:, :, m_range:-m_range, m_range:-m_range]
        loss = (pred - gt).pow(2).sum()
        # cross_entropy_loss = - (mask * torch.log(mask + 0.00001)).sum()
        # loss += 0.01 * cross_entropy_loss
        loss.backward()
        optimizer.step()
        logging.info('epoch %d, training loss: %.2f', epoch, loss.data[0])
    return model


def main():
    args = parse_args()
    logging.info(args)
    m_dict, reverse_m_dict, m_kernel = motion_dict(args.motion_range)
    m_kernel = Variable(torch.from_numpy(m_kernel).float())
    model = FullyConvNet2(args.image_size, len(m_dict))
    if torch.cuda.is_available():
        model = model.cuda()
        m_kernel = m_kernel.cuda()
    if args.train:
        if args.method == 'supervised':
            model = train_supervised(args, model, m_dict, reverse_m_dict)
        else:
            model = train_unsupervised(args, model, m_dict, reverse_m_dict, m_kernel)
        with open(os.path.join(args.save_dir, 'final.pth'), 'w') as handle:
            torch.save(model.state_dict(), handle)
    if args.test:
        model.load_state_dict(torch.load(args.init_model_path))
        test_supervised(args, model, m_dict, reverse_m_dict)

if __name__ == '__main__':
    main()
