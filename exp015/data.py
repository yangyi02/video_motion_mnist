import os
import numpy
import matplotlib.pyplot as plt
from skimage import io, transform
import h5py
import cv2

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
    if numpy.random.rand() < 0.0:
        im1, im2, im3, _ = generate_moving_images(args, images, m_dict, reverse_m_dict)
    else:
        im1, im2, im3 = generate_perspective_images(args, images, m_dict, reverse_m_dict)
    return im1, im2, im3


def generate_perspective_images(args, images, m_dict, reverse_m_dict):
    noise = 0.5
    im_size, m_range, batch_size = args.image_size, args.motion_range, args.batch_size
    im_channel = images.shape[1]
    idx = numpy.random.permutation(images.shape[0])
    im1 = images[idx[0:batch_size], :, :, :]

    max_shift = 3
    M = numpy.zeros((3, 3, batch_size))
    for i in range(batch_size):
        pts1 = numpy.float32([[0, 0], [im_size, 0], [0, im_size], [im_size, im_size]])
        shift = numpy.random.randint(-max_shift, max_shift, size=(4,2))
        pts2 = (pts1 + shift).astype(numpy.float32)
        M[:, :, i] = cv2.getPerspectiveTransform(pts1, pts2)

    bg = numpy.random.rand(batch_size, im_channel, im_size, im_size) * noise
    im2 = perspective_image(im1, M)
    im3 = perspective_image(im2, M)
    im1[im1 == 0] = bg[im1 == 0]
    im2[im2 == 0] = bg[im2 == 0]
    im3[im3 == 0] = bg[im3 == 0]
    if False:
        display(im1, im2, im3)
    return im1, im2, im3


def perspective_image(im, M):
    [batch_size, im_channel, _, im_size] = im.shape
    # Hacky code right now for quick verification
    assert im_channel == 1, 'currently hacky code only assume image channel is 1'
    im_new = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        # M = cv2.getRotationMatrix2D((im_size/2, im_size/2), angle[i], 1)
        # M = cv2.getRotationMatrix2D((0, 0), angle[i], 1)
        im_new[i, 0, :, :] = cv2.warpPerspective(im[i, :, :, :].squeeze(), M[:, :, i], (im_size, im_size))
    return im_new


def generate_moving_images(args, images, m_dict, reverse_m_dict):
    noise = 0.5
    im_size, m_range, batch_size = args.image_size, args.motion_range, args.batch_size
    im_channel = images.shape[1]
    idx = numpy.random.permutation(images.shape[0])
    im1 = images[idx[0:batch_size], :, :, :]
    m_label = numpy.random.randint(0, len(m_dict), size=batch_size)
    gt_motion = numpy.zeros((batch_size, 1, im_size, im_size))
    m_x = numpy.zeros(batch_size).astype(int)
    m_y = numpy.zeros(batch_size).astype(int)
    for i in range(batch_size):
        (m_x[i], m_y[i]) = reverse_m_dict[m_label[i]]
        gt_motion[i, :, :, :] = m_label[i]
    bg = numpy.random.rand(batch_size, im_channel, im_size, im_size) * noise
    im2 = move_image(im1, m_x, m_y)
    im3 = move_image(im2, m_x, m_y)
    gt_motion[im2 == 0] = m_dict[(0, 0)]
    im1[im1 == 0] = bg[im1 == 0]
    im2[im2 == 0] = bg[im2 == 0]
    im3[im3 == 0] = bg[im3 == 0]
    if False:
        display(im1, im2, im3, gt_motion)
    return im1, im2, im3, gt_motion.astype(int)


def move_image(im, m_x, m_y):
    [batch_size, im_channel, _, im_size] = im.shape
    m_range_x = numpy.max(numpy.abs(m_x).reshape(-1))
    m_range_y = numpy.max(numpy.abs(m_y).reshape(-1))
    m_range = max(m_range_x, m_range_y).astype(int)
    im_big = numpy.zeros((batch_size, im_channel, im_size + m_range * 2, im_size + m_range * 2))
    im_big[:, :, m_range:-m_range, m_range:-m_range] = im
    im_new = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        im_new[i, :, :, :] = im_big[i, :, m_range + m_y[i]:m_range + m_y[i] + im_size,
                             m_range + m_x[i]:m_range + m_x[i] + im_size]
    return im_new


def display(im1, im2, gt_motion):
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(im1[0, :, :, :].squeeze())
    plt.subplot(1, 2, 2)
    plt.imshow(im2[0, :, :, :].squeeze())
    plt.show()


def unit_test():
    m_dict, reverse_m_dict, m_kernel = motion_dict(1)
    args = learning_args.parse_args()
    train_images, test_images = load_mnist()
    [_, _, args.image_size, _] = train_images.shape
    generate_images(args, train_images, m_dict, reverse_m_dict)

if __name__ == '__main__':
    unit_test()
