import os
import numpy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from learning_args import parse_args
from data import generate_images, motion_dict, load_mnist
from models import FullyConvNet, FullyConvResNet, UNet, UNetBidirection
from visualize import visualize
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


def validate(args, model, images, m_dict, reverse_m_dict, m_kernel, best_test_acc):
    test_acc = test_unsupervised(args, model, images, m_dict, reverse_m_dict, m_kernel)
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
        im1, im2, im3, im4, im5, gt_motion_f, gt_motion_b = generate_images(args, images, m_dict, reverse_m_dict)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        im3 = Variable(torch.from_numpy(im3).float())
        im4 = Variable(torch.from_numpy(im4).float())
        im5 = Variable(torch.from_numpy(im5).float())
        ones = Variable(torch.ones(im3.size(0), 1, im3.size(2), im3.size(3)))
        gt_motion_f = Variable(torch.from_numpy(gt_motion_f))
        gt_motion_b = Variable(torch.from_numpy(gt_motion_b))
        if torch.cuda.is_available():
            im1, im2, im3, im4, im5 = im1.cuda(), im2.cuda(), im3.cuda(), im4.cuda(), im5.cuda()
            ones = ones.cuda()
            gt_motion_f, gt_motion_b = gt_motion_f.cuda(), gt_motion_b.cuda()
        pred, pred_f, motion_f, disappear_f, attn_f, pred_b, motion_b, disappear_b, attn_b = model(im1, im2, im4, im5, ones)
        motion_f = motion_f.transpose(1, 2).transpose(2, 3).contiguous().view(-1, model.n_class)
        gt_motion_f = gt_motion_f.contiguous().view(-1)
        motion_b = motion_b.transpose(1, 2).transpose(2, 3).contiguous().view(-1, model.n_class)
        gt_motion_b = gt_motion_b.contiguous().view(-1)
        loss_f = F.cross_entropy(motion_f, gt_motion_f)
        loss_b = F.cross_entropy(motion_b, gt_motion_b)
        loss = loss_f + loss_b
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


def test_unsupervised(args, model, images, m_dict, reverse_m_dict, m_kernel):
    m_range = args.motion_range
    test_accuracy = []
    for epoch in range(args.test_epoch):
        im1, im2, im3, im4, im5, gt_motion_f, gt_motion_b = generate_images(args, images, m_dict, reverse_m_dict)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        im3 = Variable(torch.from_numpy(im3).float())
        im4 = Variable(torch.from_numpy(im4).float())
        im5 = Variable(torch.from_numpy(im5).float())
        ones = Variable(torch.ones(im3.size(0), 1, im3.size(2), im3.size(3)))
        gt_motion_f = Variable(torch.from_numpy(gt_motion_f))
        gt_motion_b = Variable(torch.from_numpy(gt_motion_b))
        if torch.cuda.is_available():
            im1, im2, im3, im4, im5 = im1.cuda(), im2.cuda(), im3.cuda(), im4.cuda(), im5.cuda()
            ones = ones.cuda()
            gt_motion_f, gt_motion_b = gt_motion_f.cuda(), gt_motion_b.cuda()
        pred, pred_f, motion_f, disappear_f, attn_f, pred_b, motion_b, disappear_b, attn_b = model(im1, im2, im4, im5, ones)
        pred_motion_f = motion_f.max(1)[1]
        pred_motion_b = motion_b.max(1)[1]
        accuracy_f = pred_motion_f.eq(gt_motion_f).float().sum() * 1.0 / gt_motion_f.numel()
        accuracy_b = pred_motion_b.eq(gt_motion_b).float().sum() * 1.0 / gt_motion_b.numel()
        test_accuracy.append(accuracy_f.cpu().data[0])
        test_accuracy.append(accuracy_b.cpu().data[0])
        if args.display:
            m_mask_f = F.softmax(motion_f)
            flow_f = motion2flow(m_mask_f, reverse_m_dict)
            m_mask_b = F.softmax(motion_b)
            flow_b = motion2flow(m_mask_b, reverse_m_dict)
            visualize(im1, im2, im3, im4, im5, pred, flow_f, gt_motion_f, disappear_f, attn_f, flow_b, gt_motion_b, disappear_b, attn_b, m_range, reverse_m_dict)
    test_accuracy = numpy.mean(numpy.asarray(test_accuracy))
    logging.info('average testing accuracy: %.2f', test_accuracy)
    return test_accuracy


def motion2flow(m_mask, reverse_m_dict):
    [batch_size, num_class, height, width] = m_mask.size()
    kernel_x = Variable(torch.zeros(batch_size, num_class, height, width))
    kernel_y = Variable(torch.zeros(batch_size, num_class, height, width))
    if torch.cuda.is_available():
        kernel_x = kernel_x.cuda()
        kernel_y = kernel_y.cuda()
    for i in range(num_class):
        (m_x, m_y) = reverse_m_dict[i]
        kernel_x[:, i, :, :] = m_x
        kernel_y[:, i, :, :] = m_y
    flow = Variable(torch.zeros(batch_size, 2, height, width))
    flow[:, 0, :, :] = (m_mask * kernel_x).sum(1)
    flow[:, 1, :, :] = (m_mask * kernel_y).sum(1)
    return flow


def train_unsupervised(args, model, images, m_dict, reverse_m_dict, m_kernel):
    m_range = args.motion_range
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_test_acc = 0
    train_loss = []
    for epoch in range(args.train_epoch):
        optimizer.zero_grad()
        im1, im2, im3, im4, im5, gt_motion_f, gt_motion_b = generate_images(args, images, m_dict, reverse_m_dict)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        im3 = Variable(torch.from_numpy(im3).float())
        im4 = Variable(torch.from_numpy(im4).float())
        im5 = Variable(torch.from_numpy(im5).float())
        ones = Variable(torch.ones(im3.size(0), 1, im3.size(2), im3.size(3)))
        gt_motion_f = Variable(torch.from_numpy(gt_motion_f))
        gt_motion_b = Variable(torch.from_numpy(gt_motion_b))
        if torch.cuda.is_available():
            im1, im2, im3, im4, im5 = im1.cuda(), im2.cuda(), im3.cuda(), im4.cuda(), im5.cuda()
            ones = ones.cuda()
            gt_motion_f, gt_motion_b = gt_motion_f.cuda(), gt_motion_b.cuda()
        pred, pred_f, motion_f, disappear_f, attn_f, pred_b, motion_b, disappear_b, attn_b = model(im1, im2, im4, im5, ones)
        gt = im3
        # loss_f = torch.abs(pred_f - gt).sum()
        # loss_b = torch.abs(pred_b - gt).sum()
        loss = torch.abs(pred - gt).sum()
        # loss = loss + loss_f + loss_b
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
    if torch.cuda.is_available():
        m_kernel = m_kernel.cuda()
    train_images, test_images = load_mnist()
    [_, im_channel, args.image_size, _] = train_images.shape
    model = UNetBidirection(args.image_size, im_channel, len(m_dict), args.motion_range, m_kernel)
    if torch.cuda.is_available():
        model = model.cuda()
    if args.train:
        if args.method == 'supervised':
            model = train_supervised(args, model, train_images, m_dict, reverse_m_dict, m_kernel)
        else:
            model = train_unsupervised(args, model, train_images, m_dict, reverse_m_dict, m_kernel)
    if args.test:
        model.load_state_dict(torch.load(args.init_model_path))
        test_unsupervised(args, model, test_images, m_dict, reverse_m_dict, m_kernel)

if __name__ == '__main__':
    main()
