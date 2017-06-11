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
from models import FullyConvNet, FullyConvResNet, UNet, UNetU
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


def validate_supervised(args, model, images, m_dict, reverse_m_dict, m_kernel, best_test_acc):
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
        im1, im2, im3, _, _, gt_motion, _ = generate_images(args, images, m_dict, reverse_m_dict)
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
            best_test_acc = validate_supervised(args, model, images, m_dict, reverse_m_dict, m_kernel, best_test_acc)
    return model


def visualize_supervised(im1, im2, im3, pred, pred_motion, gt_motion, m_range, reverse_m_dict):
    plt.figure(1)
    plt.subplot(2,4,1)
    plt.imshow(im2[0].cpu().data.numpy().squeeze(), cmap='gray')
    plt.subplot(2,4,2)
    plt.imshow(im3[0].cpu().data.numpy().squeeze(), cmap='gray')
    plt.subplot(2,4,3)
    pred_im = pred[0].cpu().data.numpy().squeeze()
    pred_im[pred_im > 1] = 1
    pred_im[pred_im < 0] = 0
    plt.imshow(pred_im, cmap='gray')
    plt.subplot(2,4,4)
    im_diff = pred - im3
    plt.imshow(im_diff[0].cpu().data.numpy().squeeze(), cmap='gray')
    plt.subplot(2,4,6)
    gt_m = label2flow(gt_motion[0].cpu().data.numpy().squeeze(), m_range, reverse_m_dict)
    plt.imshow(gt_m)
    plt.subplot(2,4,7)
    pred_m = label2flow(pred_motion[0].cpu().data.numpy().squeeze(), m_range, reverse_m_dict)
    plt.imshow(pred_m)
    plt.show()


def label2flow(motion_label, m_range, reverse_m_dict):
    motion = numpy.zeros((motion_label.shape[0], motion_label.shape[1], 2))
    for i in range(motion_label.shape[0]):
        for j in range(motion_label.shape[1]):
            motion[i, j, :] = numpy.asarray(reverse_m_dict[motion_label[i, j]])
    mag, ang = cv2.cartToPolar(motion[..., 0], motion[..., 1])
    hsv = numpy.zeros((motion.shape[0], motion.shape[1], 3), dtype=float)
    hsv[..., 0] = ang * 180 / numpy.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = mag * 255.0 / m_range / numpy.sqrt(2)
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv.astype(numpy.uint8), cv2.COLOR_HSV2BGR)
    return rgb


def test_supervised(args, model, images, m_dict, reverse_m_dict, m_kernel):
    m_range = args.motion_range
    test_accuracy = []
    for epoch in range(args.test_epoch):
        im1, im2, im3, _, _, gt_motion, _ = generate_images(args, images, m_dict, reverse_m_dict)
        im1 = Variable(torch.from_numpy(im1).float())
        im2 = Variable(torch.from_numpy(im2).float())
        im3 = Variable(torch.from_numpy(im3).float())
        gt_motion = Variable(torch.from_numpy(gt_motion))
        if torch.cuda.is_available():
            im1, im2, im3, gt_motion = im1.cuda(), im2.cuda(), im3.cuda(), gt_motion.cuda()
        motion = model(im1, im2)
        pred_motion = motion.max(1)[1]
        accuracy = pred_motion.eq(gt_motion).float().sum() * 1.0 / gt_motion.numel()
        test_accuracy.append(accuracy.cpu().data[0])
        if args.display:
            m_range = args.motion_range
            pred = construct_image(im2, motion, m_range, m_kernel, padding=m_range)
            visualize_supervised(im1, im2, im3, pred, pred_motion, gt_motion, m_range, reverse_m_dict)
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
        pred[i, :, :, :] = F.conv2d(im_expand[i, :, :, :].unsqueeze(0), m_kernel, None, 1, padding)
    return pred


def validate_unsupervised(args, model, images, m_dict, reverse_m_dict, m_kernel, best_test_acc):
    test_acc = test_unsupervised(args, model, images, m_dict, reverse_m_dict, m_kernel)
    if test_acc >= best_test_acc:
        logging.info('model save to %s', os.path.join(args.save_dir, 'final.pth'))
        with open(os.path.join(args.save_dir, 'final.pth'), 'w') as handle:
            torch.save(model.state_dict(), handle)
        best_test_acc = test_acc
    logging.info('current best accuracy: %.2f', best_test_acc)
    return best_test_acc


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
        gt_motion_f = Variable(torch.from_numpy(gt_motion_f))
        gt_motion_b = Variable(torch.from_numpy(gt_motion_b))
        if torch.cuda.is_available():
            im1, im2, im3, im4, im5 = im1.cuda(), im2.cuda(), im3.cuda(), im4.cuda(), im5.cuda()
            gt_motion_f, gt_motion_b = gt_motion_f.cuda(), gt_motion_b.cuda()
        pred, pred_f, motion_f, pred_b, motion_b, attn_f, attn_b = model(im1, im2, im4, im5)
        # gt = im3[:, :, m_range:-m_range, m_range:-m_range]
        gt = im3
        # loss = (pred - gt).pow(2).sum()  # L1 loss is better than L2 loss
        loss_f = torch.abs(pred_f - gt).sum()
        loss_b = torch.abs(pred_b - gt).sum()
        loss = torch.abs(pred - gt).sum()
        attn_loss = torch.abs(attn_f - 0.5).sum()
        # if epoch < 1000:
        #     loss = loss + loss_f + loss_b + 0.01 * attn_loss
        # else:
        #     loss = loss + 0.01 * attn_loss
        loss = loss + loss_f + loss_b + 0.01 * attn_loss
        loss.backward()
        optimizer.step()
        train_loss.append(loss.data[0])
        if len(train_loss) > 1000:
            train_loss.pop(0)
        ave_loss = sum(train_loss) / float(len(train_loss))
        logging.info('epoch %d, training loss: %.2f, average training loss: %.2f', epoch, loss.data[0], ave_loss)
        if (epoch+1) % args.test_interval == 0:
            logging.info('epoch %d, testing', epoch)
            best_test_acc = validate_unsupervised(args, model, images, m_dict, reverse_m_dict, m_kernel, best_test_acc)
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
        gt_motion_f = Variable(torch.from_numpy(gt_motion_f))
        gt_motion_b = Variable(torch.from_numpy(gt_motion_b))
        if torch.cuda.is_available():
            im1, im2, im3, im4, im5 = im1.cuda(), im2.cuda(), im3.cuda(), im4.cuda(), im5.cuda()
            gt_motion_f, gt_motion_b = gt_motion_f.cuda(), gt_motion_b.cuda()
        pred, pred_f, motion_f, pred_b, motion_b, attn_f, attn_b = model(im1, im2, im4, im5)
        pred_motion_f = motion_f.max(1)[1]
        accuracy_f = pred_motion_f.eq(gt_motion_f).float().sum() * 1.0 / gt_motion_f.numel()
        pred_motion_b = motion_b.max(1)[1]
        accuracy_b = pred_motion_b.eq(gt_motion_b).float().sum() * 1.0 / gt_motion_b.numel()
        test_accuracy.append(accuracy_f.cpu().data[0])
        test_accuracy.append(accuracy_b.cpu().data[0])
        if args.display:
            m_range = args.motion_range
            visualize_unsupervised(im1, im2, im3, im4, im5, pred, pred_motion_f, gt_motion_f, attn_f, pred_motion_b, gt_motion_b, attn_b, m_range, reverse_m_dict)
    test_accuracy = numpy.mean(numpy.asarray(test_accuracy))
    logging.info('average testing accuracy: %.2f', test_accuracy)
    return test_accuracy


def visualize_unsupervised(im1, im2, im3, im4, im5, pred, pred_motion_f, gt_motion_f, attn_f, pred_motion_b, gt_motion_b, attn_b, m_range, reverse_m_dict):
    plt.figure(1)
    plt.subplot(4,3,1)
    plt.imshow(im2[0].cpu().data.numpy().squeeze(), cmap='gray')
    plt.subplot(4,3,2)
    plt.imshow(im3[0].cpu().data.numpy().squeeze(), cmap='gray')
    plt.subplot(4,3,3)
    plt.imshow(im4[0].cpu().data.numpy().squeeze(), cmap='gray')
    plt.subplot(4,3,5)
    pred_im = pred[0].cpu().data.numpy().squeeze()
    pred_im[pred_im > 1] = 1
    pred_im[pred_im < 0] = 0
    plt.imshow(pred_im, cmap='gray')
    plt.subplot(4,3,6)
    im_diff = pred - im3
    plt.imshow(im_diff[0].cpu().data.numpy().squeeze(), cmap='gray')
    plt.subplot(4,3,7)
    gt_m_f = label2flow(gt_motion_f[0].cpu().data.numpy().squeeze(), m_range, reverse_m_dict)
    plt.imshow(gt_m_f)
    plt.subplot(4,3,8)
    pred_m_f = label2flow(pred_motion_f[0].cpu().data.numpy().squeeze(), m_range, reverse_m_dict)
    plt.imshow(pred_m_f)
    plt.subplot(4,3,9)
    plt.imshow(attn_f[0].cpu().data.numpy().squeeze())
    plt.subplot(4,3,10)
    gt_m_b = label2flow(gt_motion_b[0].cpu().data.numpy().squeeze(), m_range, reverse_m_dict)
    plt.imshow(gt_m_b)
    plt.subplot(4,3,11)
    pred_m_b = label2flow(pred_motion_b[0].cpu().data.numpy().squeeze(), m_range, reverse_m_dict)
    plt.imshow(pred_m_b)
    plt.subplot(4,3,12)
    plt.imshow(attn_b[0].cpu().data.numpy().squeeze())
    plt.show()


def main():
    args = parse_args()
    logging.info(args)
    m_dict, reverse_m_dict, m_kernel = motion_dict(args.motion_range)
    m_kernel = Variable(torch.from_numpy(m_kernel).float())
    if torch.cuda.is_available():
        m_kernel = m_kernel.cuda()
    train_images, test_images = load_mnist()
    [_, im_channel, args.image_size, _] = train_images.shape
    if args.method == 'supervised':
        model = UNet(args.image_size, im_channel, len(m_dict))
    else:
        model = UNetU(args.image_size, im_channel, len(m_dict), args.motion_range, m_kernel)
    if torch.cuda.is_available():
        model = model.cuda()
    if args.train:
        if args.method == 'supervised':
            model = train_supervised(args, model, train_images, m_dict, reverse_m_dict, m_kernel)
        else:
            model = train_unsupervised(args, model, train_images, m_dict, reverse_m_dict, m_kernel)
    if args.test:
        model.load_state_dict(torch.load(args.init_model_path))
        if args.method == 'supervised':
            test_supervised(args, model, test_images, m_dict, reverse_m_dict, m_kernel)
        else:
            test_unsupervised(args, model, test_images, m_dict, reverse_m_dict, m_kernel)

if __name__ == '__main__':
    main()
