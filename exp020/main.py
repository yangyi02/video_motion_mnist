import os
import numpy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from learning_args import parse_args
from data import motion_dict, get_robot_meta, load_robot_data, generate_batch
from models import FullyConvNet, FullyConvResNet, UNet
from visualize import visualize
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


def validate(args, model, m_dict, reverse_m_dict, m_kernel, best_test_loss):
    test_loss = test_unsupervised(args, model, m_dict, reverse_m_dict, m_kernel)
    if test_loss <= best_test_loss:
        logging.info('model save to %s', os.path.join(args.save_dir, 'final.pth'))
        with open(os.path.join(args.save_dir, 'final.pth'), 'w') as handle:
            torch.save(model.state_dict(), handle)
        best_test_loss = test_loss
    logging.info('current best accuracy: %.2f', best_test_loss)
    return best_test_loss


def test_unsupervised(args, model, m_dict, reverse_m_dict, m_kernel):
    robot_meta, robot_dir = get_robot_meta()
    m_range = args.motion_range
    test_loss = []
    for epoch in range(args.test_epoch):
        images = load_robot_data(args, robot_meta, robot_dir)
        for i in range(1, 24):
            im1, im2, im3 = generate_batch(args, images)
            im1 = Variable(torch.from_numpy(im1).float())
            im2 = Variable(torch.from_numpy(im2).float())
            im3 = Variable(torch.from_numpy(im3).float())
            if torch.cuda.is_available():
                im1, im2, im3 = im1.cuda(), im2.cuda(), im3.cuda()
            motion = model(im1, im2)
            pred = construct_image(im2, motion, m_range, m_kernel, padding=0)
            gt = im3[:, :, m_range:-m_range, m_range:-m_range]
            # loss = (pred - gt).pow(2).sum()  # L1 loss is better than L2 loss
            loss = torch.abs(pred - gt).sum()
            if args.display:
                m_range = args.motion_range
                pred = construct_image(im2, motion, m_range, m_kernel, padding=m_range)
                pred_motion = motion.max(1)[1]
                visualize(im1, im2, im3, pred, pred_motion, F.softmax(motion), m_range, m_dict, reverse_m_dict)
            test_loss.append(loss.data[0])
    test_loss = numpy.mean(numpy.asarray(test_loss))
    logging.info('average testing loss: %.2f', test_loss)
    return test_loss


def construct_image(im, motion, m_range, m_kernel, padding=0):
    mask = F.softmax(motion)
    height = im.size(2) - 2 * m_range + 2 * padding
    width = im.size(3) - 2 * m_range + 2 * padding
    im_channel = im.size(1)
    pred = Variable(torch.Tensor(im.size(0), im_channel, height, width))
    if torch.cuda.is_available():
        pred = pred.cuda()
    for i in range(im_channel):
        im_expand = im[:, i, :, :].unsqueeze(1).expand_as(mask) * mask
        for j in range(im.size(0)):
            pred[j, i, :, :] = F.conv2d(im_expand[j, :-1, :, :].unsqueeze(0), m_kernel, None, 1, padding)
    return pred


def train_unsupervised(args, model, m_dict, reverse_m_dict, m_kernel):
    robot_meta, robot_dir = get_robot_meta()
    m_range = args.motion_range
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    train_loss = []
    best_test_loss = 1e10
    for epoch in range(args.train_epoch):
        images = load_robot_data(args, robot_meta, robot_dir)
        current_train_loss = []
        for i in range(1, 24):
            optimizer.zero_grad()
            im1, im2, im3 = generate_batch(args, images)
            im1 = Variable(torch.from_numpy(im1).float())
            im2 = Variable(torch.from_numpy(im2).float())
            im3 = Variable(torch.from_numpy(im3).float())
            if torch.cuda.is_available():
                im1, im2, im3 = im1.cuda(), im2.cuda(), im3.cuda()
            motion = model(im1, im2)
            pred = construct_image(im2, motion, m_range, m_kernel, padding=0)
            gt = im3[:, :, m_range:-m_range, m_range:-m_range]
            # loss = (pred - gt).pow(2).sum()  # L1 loss is better than L2 loss
            loss = torch.abs(pred - gt).sum()
            loss.backward()
            optimizer.step()
            current_train_loss.append(loss.data[0])
        current_ave_loss = sum(current_train_loss) / float(len(current_train_loss))
        train_loss.append(current_ave_loss)
        if len(train_loss) > 1000:
            train_loss.pop(0)
        ave_loss = sum(train_loss) / float(len(train_loss))
        logging.info('epoch %d, training loss: %.2f, average training loss: %.2f', epoch, current_ave_loss, ave_loss)
        if (epoch+1) % args.test_interval == 0:
            logging.info('epoch %d, testing', epoch)
            best_test_loss = validate(args, model, m_dict, reverse_m_dict, m_kernel, best_test_loss)
    return model


def main():
    args = parse_args()
    logging.info(args)
    m_dict, reverse_m_dict, m_kernel = motion_dict(args.motion_range)
    m_kernel = Variable(torch.from_numpy(m_kernel).float())
    robot_meta, robot_dir = get_robot_meta()
    images = load_robot_data(args, robot_meta, robot_dir)
    im1, im2, im3 = generate_batch(args, images)
    [_, im_channel, args.image_height, args.image_width] = im1.shape
    model = FullyConvNet(args.image_height, im_channel, len(m_dict) + 1)
    if torch.cuda.is_available():
        model = model.cuda()
        m_kernel = m_kernel.cuda()
    if args.train:
        model = train_unsupervised(args, model, m_dict, reverse_m_dict, m_kernel)
    if args.test:
        model.load_state_dict(torch.load(args.init_model_path))
        test_unsupervised(args, model, m_dict, reverse_m_dict, m_kernel)

if __name__ == '__main__':
    main()
