import numpy
import cv2
import matplotlib.pyplot as plt
import flowlib


def visualize(im1, im2, pred, pred_motion, gt_motion, disappear, m_range, reverse_m_dict):
    img_size = im1.size(2)
    width, height = get_img_size(2, 5, img_size)
    img = numpy.ones((height, width, 3))

    im1 = im1[0].cpu().data.numpy().transpose(1, 2, 0).repeat(3, 2)
    x1, y1, x2, y2 = get_img_coordinate(1, 2, img_size)
    img[y1:y2, x1:x2, :] = im1

    im2 = im2[0].cpu().data.numpy().transpose(1, 2, 0).repeat(3, 2)
    x1, y1, x2, y2 = get_img_coordinate(1, 3, img_size)
    img[y1:y2, x1:x2, :] = im2

    im_diff = numpy.abs(im1 - im2)
    x1, y1, x2, y2 = get_img_coordinate(1, 4, img_size)
    img[y1:y2, x1:x2, :] = im_diff

    pred_motion = pred_motion[0].cpu().data.numpy().transpose(1, 2, 0)
    optical_flow = flowlib.visualize_flow(pred_motion, m_range)
    x1, y1, x2, y2 = get_img_coordinate(2, 1, img_size)
    img[y1:y2, x1:x2, :] = optical_flow / 255.0

    gt_motion = label2motion(gt_motion[0].cpu().data.numpy().squeeze(), reverse_m_dict)
    optical_flow = flowlib.visualize_flow(gt_motion, m_range)
    x1, y1, x2, y2 = get_img_coordinate(2, 2, img_size)
    img[y1:y2, x1:x2, :] = optical_flow / 255.0

    pred = pred[0].cpu().data.numpy().transpose(1, 2, 0).repeat(3, 2)
    pred[pred > 1] = 1
    pred[pred < 0] = 0
    x1, y1, x2, y2 = get_img_coordinate(2, 3, img_size)
    img[y1:y2, x1:x2, :] = pred

    im_diff = numpy.abs(pred - im2)
    x1, y1, x2, y2 = get_img_coordinate(2, 4, img_size)
    img[y1:y2, x1:x2, :] = im_diff

    disappear = disappear[0].cpu().data.numpy().transpose(1, 2, 0).repeat(3, 2)
    x1, y1, x2, y2 = get_img_coordinate(2, 5, img_size)
    img[y1:y2, x1:x2, :] = disappear

    plt.figure(1)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def get_img_size(n_row, n_col, img_size):
    height = n_row * img_size + (n_row - 1) * int(img_size/10)
    width = n_col * img_size + (n_col - 1) * int(img_size/10)
    return width, height


def get_img_coordinate(row, col, img_size):
    y1 = (row - 1) * img_size + (row - 1) * int(img_size/10)
    y2 = y1 + img_size
    x1 = (col - 1) * img_size + (col - 1) * int(img_size/10)
    x2 = x1 + img_size
    return x1, y1, x2, y2


def label2motion(motion_label, reverse_m_dict):
    motion = numpy.zeros((motion_label.shape[0], motion_label.shape[1], 2))
    for i in range(motion_label.shape[0]):
        for j in range(motion_label.shape[1]):
            motion[i, j, :] = numpy.asarray(reverse_m_dict[motion_label[i, j]])
    return motion


def motion2color(motion, m_range):
    mag, ang = cv2.cartToPolar(motion[..., 0], motion[..., 1])
    hsv = numpy.zeros((motion.shape[0], motion.shape[1], 3), dtype=float)
    hsv[..., 0] = ang * 180 / numpy.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = mag * 255.0 / m_range / numpy.sqrt(2)
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv.astype(numpy.uint8), cv2.COLOR_HSV2BGR)
    return rgb

