import os
import numpy
import matplotlib.pyplot as plt
from skimage import io, transform
import pickle
from PIL import Image

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


def get_mpii_meta(meta_file='../data/mpii/mpii_meta.pkl', mpii_dir='../data/mpii/mpii-128'):
    mpii_meta = pickle.load(open(meta_file))
    n = 4  # use 2 images as inputs and 1 image as output, hence need at least 3 images for training
    new_mpii_meta = {}
    cnt = 0
    for k, v in mpii_meta.iteritems():
        image_names = v[2]
        for i in range(len(image_names)):
            image_names[i] = os.path.join(mpii_dir, v[0], v[1], image_names[i])
        num_images = len(image_names)
        idx = range(0, num_images - n)
        for i in range(len(idx)):
            start_idx = idx[i]
            end_idx = idx[i] + n
            new_mpii_meta[cnt] = image_names[start_idx:end_idx]
            cnt += 1
    return new_mpii_meta


def generate_batch(args, mpii_meta):
    batch_size, height, width, im_channel = args.batch_size, 128, 128, 3
    idx = numpy.random.permutation(len(mpii_meta))[0:batch_size]
    n = 4
    im_input = numpy.zeros((batch_size, im_channel * (n - 1), height, width))
    im_output = numpy.zeros((batch_size, im_channel, height, width))
    for i in range(batch_size):
        image_names = mpii_meta[idx[i]]
        n_image = len(image_names)
        assert n_image == n
        for j in range(n_image):
            image_name = image_names[j]
            im = numpy.array(Image.open(image_name)) / 255.0
            im = im.transpose((2, 0, 1))
            if j == 0:
                _, im_height, im_width = im.shape
                idx_h = numpy.random.randint(0, im_height+1-height)
                idx_w = numpy.random.randint(0, im_width+1-width)
            if j < n_image - 1:
                im_input[i, j*im_channel:(j+1)*im_channel, :, :] = im[:, idx_h:idx_h+height, idx_w:idx_w+width]
            else:
                im_output[i, :, :, :] = im[:, idx_h:idx_h+height, idx_w:idx_w+width]
    return im_input, im_output


def display(images1, images2, images3):
    for i in range(images1.shape[0]):
        plt.figure(1)
        plt.subplot(2, 3, 1)
        im1 = images1[i, :, :, :].squeeze().transpose(1, 2, 0)
        plt.imshow(im1)
        plt.subplot(2, 3, 2)
        im2 = images2[i, :, :, :].squeeze().transpose(1, 2, 0)
        plt.imshow(im2)
        plt.subplot(2, 3, 3)
        im3 = images3[i, :, :, :].squeeze().transpose(1, 2, 0)
        plt.imshow(im3)
        plt.subplot(2, 3, 4)
        im_diff1 = abs(im2 - im1)
        plt.imshow(im_diff1)
        plt.subplot(2, 3, 5)
        im_diff2 = abs(im3 - im2)
        plt.imshow(im_diff2)
        plt.show()


def unit_test():
    m_dict, reverse_m_dict, m_kernel = motion_dict(1)
    args = learning_args.parse_args()
    mpii_meta = get_mpii_meta()
    im_input, im_output = generate_batch(args, mpii_meta)
    if True:
        im1 = im_input[:, 0:3, :, :]
        im2 = im_input[:, 3:6, :, :]
        im3 = im_output
        display(im1, im2, im3)

if __name__ == '__main__':
    unit_test()
