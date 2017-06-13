import os
import numpy
import matplotlib.pyplot as plt
from skimage import io, transform
import h5py

import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


def motion_dict(m_range):
    m_dict, reverse_m_dict = {}, {}
    x = numpy.linspace(-m_range, m_range, 2 * m_range + 1)
    y = numpy.linspace(-m_range, m_range, 2 * m_range + 1)
    m_x, m_y = numpy.meshgrid(x, y)
    m_x, m_y, = m_x.reshape(-1).astype(int), m_y.reshape(-1).astype(int)
    m_kernel = numpy.zeros((1, len(m_x), 2 * m_range + 1, 2 * m_range + 1))
    for i in range(len(m_x)):
        m_dict[(m_x[i], m_y[i])] = i
        reverse_m_dict[i] = (m_x[i], m_y[i])
        m_kernel[:, i, m_y[i] + m_range, m_x[i] + m_range] = 1
    return m_dict, reverse_m_dict, m_kernel


def load_mnist(file_name='../mnist.h5'):
    script_dir = os.path.dirname(__file__)  # absolute dir the script is in
    f = h5py.File(os.path.join(script_dir, file_name))
    train_images = f['train'].value.reshape(-1, 28, 28)
    train_images = numpy.pad(train_images, ((0, 0), (2, 2), (2, 2)), 'constant')
    train_images = numpy.expand_dims(train_images, 1)
    test_images = f['test'].value.reshape(-1, 28, 28)
    test_images = numpy.pad(test_images, ((0, 0), (2, 2), (2, 2)), 'constant')
    test_images = numpy.expand_dims(test_images, 1)
    return train_images, test_images


def generate_images(args, images, m_dict, reverse_m_dict):
    noise = 0.5
    im_size, m_range, batch_size = args.image_size, args.motion_range, args.batch_size
    im_channel = images.shape[1]
    idx = numpy.random.permutation(images.shape[0])
    im1_3 = images[idx[0:batch_size], :, :, :]
    idx = numpy.random.permutation(images.shape[0])
    im2_3 = images[idx[0:batch_size], :, :, :]
    bg3 = numpy.random.rand(batch_size, im_channel, im_size, im_size) * noise
    m_label = numpy.random.randint(0, len(m_dict), size=(batch_size, 3))
    m_f_x = numpy.zeros((batch_size, 3)).astype(int)
    m_f_y = numpy.zeros((batch_size, 3)).astype(int)
    m_b_x = numpy.zeros((batch_size, 3)).astype(int)
    m_b_y = numpy.zeros((batch_size, 3)).astype(int)
    for i in range(batch_size):
        for j in range(m_label.shape[1]):
            (m_f_x[i, j], m_f_y[i, j]) = reverse_m_dict[m_label[i, j]]
            (m_b_x[i, j], m_b_y[i, j]) = (-m_f_x[i, j], -m_f_y[i, j])
    motion_f_1 = numpy.zeros((batch_size, 1, im_size, im_size))
    motion_b_1 = numpy.zeros((batch_size, 1, im_size, im_size))
    motion_f_2 = numpy.zeros((batch_size, 1, im_size, im_size))
    motion_b_2 = numpy.zeros((batch_size, 1, im_size, im_size))
    bg_motion_f = numpy.zeros((batch_size, 1, im_size, im_size))
    bg_motion_b = numpy.zeros((batch_size, 1, im_size, im_size))
    for i in range(batch_size):
        motion_f_1[i, :, :, :] = m_label[i, 0]
        motion_b_1[i, :, :, :] = m_dict[(m_b_x[i, 0], m_b_y[i, 0])]
        motion_f_2[i, :, :, :] = m_label[i, 1]
        motion_b_2[i, :, :, :] = m_dict[(m_b_x[i, 1], m_b_y[i, 1])]
        bg_motion_f[i, :, :, :] = m_label[i, 2]
        bg_motion_b[i, :, :, :] = m_dict[(m_b_x[i, 2], m_b_y[i, 2])]
    d = 7
    shift = numpy.random.randint(-d, d, size=(batch_size, 2))
    im1_3 = move_image_fg(im1_3, shift[:, 0], shift[:, 1], d)
    im2_3 = move_image_fg(im2_3, -shift[:, 0], -shift[:, 1], d)
    im1_2 = move_image_fg(im1_3, m_b_x[:, 0], m_b_y[:, 0], m_range)
    im1_1 = move_image_fg(im1_2, m_b_x[:, 0], m_b_y[:, 0], m_range)
    im1_4 = move_image_fg(im1_3, m_f_x[:, 0], m_f_y[:, 0], m_range)
    im1_5 = move_image_fg(im1_4, m_f_x[:, 0], m_f_y[:, 0], m_range)
    im2_2 = move_image_fg(im2_3, m_b_x[:, 1], m_b_y[:, 1], m_range)
    im2_1 = move_image_fg(im2_2, m_b_x[:, 1], m_b_y[:, 1], m_range)
    im2_4 = move_image_fg(im2_3, m_f_x[:, 1], m_f_y[:, 1], m_range)
    im2_5 = move_image_fg(im2_4, m_f_x[:, 1], m_f_y[:, 1], m_range)
    bg2 = move_image_bg(bg3, m_b_x[:, 2], m_b_y[:, 2], m_range)
    bg1 = move_image_bg(bg2, m_b_x[:, 2], m_b_y[:, 2], m_range)
    bg4 = move_image_bg(bg3, m_f_x[:, 2], m_f_y[:, 2], m_range)
    bg5 = move_image_bg(bg4, m_f_x[:, 2], m_f_y[:, 2], m_range)
    for i in range(batch_size):
        motion_f_2[im2_2 == 0] = bg_motion_f[im2_2 == 0]
        motion_b_2[im2_4 == 0] = bg_motion_b[im2_4 == 0]
    im2_1[im2_1 == 0] = bg1[im2_1 == 0]
    im2_2[im2_2 == 0] = bg2[im2_2 == 0]
    im2_3[im2_3 == 0] = bg3[im2_3 == 0]
    im2_4[im2_4 == 0] = bg4[im2_4 == 0]
    im2_5[im2_5 == 0] = bg5[im2_5 == 0]
    for i in range(batch_size):
        motion_f_1[im1_2 == 0] = motion_f_2[im1_2 == 0]
        motion_b_1[im1_4 == 0] = motion_b_2[im1_4 == 0]
    im1_1[im1_1 == 0] = im2_1[im1_1 == 0]
    im1_2[im1_2 == 0] = im2_2[im1_2 == 0]
    im1_3[im1_3 == 0] = im2_3[im1_3 == 0]
    im1_4[im1_4 == 0] = im2_4[im1_4 == 0]
    im1_5[im1_5 == 0] = im2_5[im1_5 == 0]
    if False:
        display(im1_1, im1_2, im1_3, im1_4, im1_5, motion_f_1, motion_b_1)
    return im1_1, im1_2, im1_3, im1_4, im1_5, motion_f_1.astype(int), motion_b_1.astype(int)


def move_image_fg(im, m_x, m_y, m_range):
    [batch_size, im_channel, _, im_size] = im.shape
    im_big = numpy.zeros((batch_size, im_channel, im_size + m_range * 2, im_size + m_range * 2))
    im_big[:, :, m_range:-m_range, m_range:-m_range] = im
    im_new = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        im_new[i, :, :, :] = im_big[i, :, m_range + m_y[i]:m_range + m_y[i] + im_size,
                             m_range + m_x[i]:m_range + m_x[i] + im_size]
    return im_new


def move_image_bg(im, m_x, m_y, m_range):
    noise = 0.5
    [batch_size, im_channel, _, im_size] = im.shape
    im_big = numpy.random.rand(batch_size, im_channel, im_size + m_range * 2, im_size + m_range * 2) * noise
    im_big[:, :, m_range:-m_range, m_range:-m_range] = im
    im_new = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        im_new[i, :, :, :] = im_big[i, :, m_range + m_y[i]:m_range + m_y[i] + im_size,
                             m_range + m_x[i]:m_range + m_x[i] + im_size]
    return im_new


def display(im1, im2, im3, im4, im5, gt_motion_f, gt_motion_b):
    plt.figure(1)
    plt.subplot(1, 5, 1)
    plt.imshow(im1[0, :, :, :].squeeze())
    plt.axis('off')
    plt.subplot(1, 5, 2)
    plt.imshow(im2[0, :, :, :].squeeze())
    plt.axis('off')
    plt.subplot(1, 5, 3)
    plt.imshow(im3[0, :, :, :].squeeze())
    plt.axis('off')
    plt.subplot(1, 5, 4)
    plt.imshow(im4[0, :, :, :].squeeze())
    plt.axis('off')
    plt.subplot(1, 5, 5)
    plt.imshow(im5[0, :, :, :].squeeze())
    plt.axis('off')
    plt.show()


def unit_test():
    m_dict, reverse_m_dict, m_kernel = motion_dict(1)
    args = learning_args.parse_args()
    train_images, test_images = load_mnist()
    [_, _, args.image_size, _] = train_images.shape
    generate_images(args, train_images, m_dict, reverse_m_dict)

if __name__ == '__main__':
    unit_test()
