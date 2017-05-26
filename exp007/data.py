import os
import numpy
from skimage import io, transform
import matplotlib.pyplot as plt
import h5py


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
    train_images = numpy.expand_dims(train_images, 1)
    test_images = f['test'].value.reshape(-1, 28, 28)
    test_images = numpy.expand_dims(test_images, 1)
    return train_images, test_images


def generate_images(args, images, m_dict, reverse_m_dict):
    noise_magnitude = 0.2
    im_size, m_range, batch_size = args.image_size, args.motion_range, args.batch_size
    im_channel = images.shape[1]
    idx = numpy.random.permutation(images.shape[0])
    im1 = images[idx[0:batch_size], :, :, :]
    bg1 = numpy.random.rand(batch_size, im_channel, im_size, im_size) * noise_magnitude
    im_big = numpy.zeros((batch_size, im_channel, im_size + m_range * 2, im_size + m_range * 2))
    im_big[:, :, m_range:-m_range, m_range:-m_range] = im1
    im2 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    m_label = numpy.random.randint(0, len(m_dict), size=batch_size)
    gt_motion = numpy.zeros((batch_size, 1, im_size, im_size))
    for i in range(batch_size):
        (m_x, m_y) = reverse_m_dict[m_label[i]]
        im2[i, :, :, :] = im_big[i, :, m_range + m_y:m_range + m_y + im_size,
                          m_range + m_x:m_range + m_x + im_size]
        gt_motion[i, :, :, :] = m_label[i]
    bg_big = numpy.random.rand(batch_size, im_channel, im_size + m_range * 2,
                               im_size + m_range * 2) * noise_magnitude
    bg_big[:, :, m_range:-m_range, m_range:-m_range] = bg1
    bg2 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    bg_m_label = numpy.random.randint(0, len(m_dict), size=batch_size)
    bg_gt_motion = numpy.zeros((batch_size, 1, im_size, im_size))
    for i in range(batch_size):
        (bg_m_x, bg_m_y) = reverse_m_dict[bg_m_label[i]]
        bg2[i, :, :, :] = bg_big[i, :, m_range + bg_m_y:m_range + bg_m_y + im_size,
                          m_range + bg_m_x:m_range + bg_m_x + im_size]
        bg_gt_motion[i, :, :, :] = bg_m_label[i]

    im_big = numpy.zeros((batch_size, im_channel, im_size + m_range * 2, im_size + m_range * 2))
    im_big[:, :, m_range:-m_range, m_range:-m_range] = im2
    im3 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (m_x, m_y) = reverse_m_dict[m_label[i]]
        im3[i, :, :, :] = im_big[i, :, m_range + m_y:m_range + m_y + im_size,
                          m_range + m_x:m_range + m_x + im_size]
    bg_big = numpy.random.rand(batch_size, im_channel, im_size + m_range * 2,
                               im_size + m_range * 2) * noise_magnitude
    bg_big[:, :, m_range:-m_range, m_range:-m_range] = bg2
    bg3 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (bg_m_x, bg_m_y) = reverse_m_dict[bg_m_label[i]]
        bg3[i, :, :, :] = bg_big[i, :, m_range + bg_m_y:m_range + bg_m_y + im_size,
                          m_range + bg_m_x:m_range + bg_m_x + im_size]

    gt_motion[im2 == 0] = bg_gt_motion[im2 == 0]
    im1[im1 == 0] = bg1[im1 == 0]
    im2[im2 == 0] = bg2[im2 == 0]
    im3[im3 == 0] = bg3[im3 == 0]

    if args.display:
        plt.figure(4)
        plt.subplot(2,2,1)
        plt.imshow(im1[0, :, :, :].squeeze())
        plt.subplot(2,2,2)
        plt.imshow(im2[0, :, :, :].squeeze())
        plt.subplot(2,2,3)
        plt.imshow(im3[0, :, :, :].squeeze())
        plt.subplot(2,2,4)
        plt.imshow(gt_motion[0, :, :, :].squeeze())
        plt.show()

    return im1, im2, im3, gt_motion.astype(int)
