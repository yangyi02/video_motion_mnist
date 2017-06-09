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
    im1_1 = images[idx[0:batch_size], :, :, :]
    bg1_1 = numpy.random.rand(batch_size, im_channel, im_size, im_size) * noise_magnitude
    idx = numpy.random.permutation(images.shape[0])
    im1_2 = images[idx[0:batch_size], :, :, :]
    bg1_2 = numpy.random.rand(batch_size, im_channel, im_size, im_size) * noise_magnitude

    im_big = numpy.zeros((batch_size, im_channel, im_size+m_range*2, im_size+m_range*2))
    im_big[:, :, m_range:-m_range, m_range:-m_range] = im1_1
    im2_1 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    motion_label1 = numpy.random.randint(0, len(m_dict), size=batch_size)
    gt_motion1 = numpy.zeros((batch_size, 1, im_size, im_size))
    for i in range(batch_size):
        (m_x, m_y) = reverse_m_dict[motion_label1[i]]
        im2_1[i, :, :, :] = im_big[i, :, m_range+m_y:m_range+m_y+im_size,
              m_range+m_x:m_range+m_x+im_size]
        gt_motion1[i, :, :, :] = motion_label1[i]
    im_big = numpy.zeros((batch_size, im_channel, im_size + m_range * 2, im_size + m_range * 2))
    im_big[:, :, m_range:-m_range, m_range:-m_range] = im1_2
    im2_2 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    motion_label2 = numpy.random.randint(0, len(m_dict), size=batch_size)
    gt_motion2 = numpy.zeros((batch_size, 1, im_size, im_size))
    for i in range(batch_size):
        (m_x, m_y) = reverse_m_dict[motion_label2[i]]
        im2_2[i, :, :, :] = im_big[i, :, m_range + m_y:m_range + m_y + im_size,
                            m_range + m_x:m_range + m_x + im_size]
        gt_motion2[i, :, :, :] = motion_label2[i]
    bg_motion_label = numpy.random.randint(0, len(m_dict), size=batch_size)
    bg_gt_motion1 = numpy.zeros((batch_size, 1, im_size, im_size))
    bg_big = numpy.random.rand(batch_size, im_channel, im_size+m_range*2, im_size+m_range*2) * noise_magnitude
    bg_big[:, :, m_range:-m_range, m_range:-m_range] = bg1_1
    bg2_1 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (bg_m_x, bg_m_y) = reverse_m_dict[bg_motion_label[i]]
        bg2_1[i, :, :, :] = bg_big[i, :, m_range+bg_m_y:m_range+bg_m_y+im_size,
              m_range+bg_m_x:m_range+bg_m_x+im_size]
        bg_gt_motion1[i, :, :, :] = bg_motion_label[i]
    bg_gt_motion2 = numpy.zeros((batch_size, 1, im_size, im_size))
    bg_big = numpy.random.rand(batch_size, im_channel, im_size + m_range * 2,
                                im_size + m_range * 2) * noise_magnitude
    bg_big[:, :, m_range:-m_range, m_range:-m_range] = bg1_2
    bg2_2 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (bg_m_x, bg_m_y) = reverse_m_dict[bg_motion_label[i]]
        bg2_2[i, :, :, :] = bg_big[i, :, m_range + bg_m_y:m_range + bg_m_y + im_size,
                            m_range + bg_m_x:m_range + bg_m_x + im_size]
        bg_gt_motion2[i, :, :, :] = bg_motion_label[i]

    im_big = numpy.zeros((batch_size, im_channel, im_size+m_range*2, im_size+m_range*2))
    im_big[:, :, m_range:-m_range, m_range:-m_range] = im2_1
    im3_1 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (m_x, m_y) = reverse_m_dict[motion_label1[i]]
        im3_1[i, :, :, :] = im_big[i, :, m_range + m_y:m_range+m_y+im_size,
                          m_range+m_x:m_range+m_x+im_size]
    im_big = numpy.zeros((batch_size, im_channel, im_size+m_range*2, im_size+m_range*2))
    im_big[:, :, m_range:-m_range, m_range:-m_range] = im2_2
    im3_2 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (m_x, m_y) = reverse_m_dict[motion_label1[i]]
        im3_2[i, :, :, :] = im_big[i, :, m_range + m_y:m_range+m_y+im_size,
                          m_range+m_x:m_range+m_x+im_size]
    bg_big = numpy.random.rand(batch_size, im_channel, im_size+m_range*2, im_size+m_range*2) * noise_magnitude
    bg_big[:, :, m_range:-m_range, m_range:-m_range] = bg2_1
    bg3_1 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (bg_m_x, bg_m_y) = reverse_m_dict[bg_motion_label[i]]
        bg3_1[i, :, :, :] = bg_big[i, :, m_range+bg_m_y:m_range+bg_m_y+im_size,
              m_range+bg_m_x:m_range+bg_m_x+im_size]
    bg_big = numpy.random.rand(batch_size, im_channel, im_size + m_range * 2,
                               im_size + m_range * 2) * noise_magnitude
    bg_big[:, :, m_range:-m_range, m_range:-m_range] = bg2_2
    bg3_2 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    for i in range(batch_size):
        (bg_m_x, bg_m_y) = reverse_m_dict[bg_motion_label[i]]
        bg3_2[i, :, :, :] = bg_big[i, :, m_range + bg_m_y:m_range + bg_m_y + im_size,
                            m_range + bg_m_x:m_range + bg_m_x + im_size]

    gt_motion1[im2_1 == 0] = bg_gt_motion1[im2_1 == 0]
    gt_motion2[im2_2 == 0] = bg_gt_motion2[im2_2 == 0]
    im1_1[im1_1 == 0] = bg1_1[im1_1 == 0]
    im1_2[im1_2 == 0] = bg1_2[im1_2 == 0]
    im2_1[im2_1 == 0] = bg2_1[im2_1 == 0]
    im2_2[im2_2 == 0] = bg2_2[im2_2 == 0]
    im3_1[im3_1 == 0] = bg3_1[im3_1 == 0]
    im3_2[im3_2 == 0] = bg3_2[im3_2 == 0]
    im1 = numpy.concatenate((im1_1, im1_2), 3)
    im2 = numpy.concatenate((im2_1, im2_2), 3)
    im3 = numpy.concatenate((im3_1, im3_2), 3)
    gt_motion = numpy.concatenate((gt_motion1, gt_motion2), 3)
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
