import os
import numpy
from skimage import io, transform
import matplotlib.pyplot as plt
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
    for i in range(len(m_x)):
        m_dict[(m_x[i], m_y[i])] = i
        reverse_m_dict[i] = (m_x[i], m_y[i])
    return m_dict, reverse_m_dict


def load_images(im_size, image_dir='images'):
    script_dir = os.path.dirname(__file__)  # absolute dir the script is in
    image_files = os.listdir(os.path.join(script_dir, image_dir))
    images = []
    for image_file in image_files:
        if image_file.endswith('.jpg'):
            print('loading %s' % image_file)
            image = io.imread(os.path.join(script_dir, image_dir, image_file))
            image = transform.resize(image, (im_size, im_size), mode='constant')
            images.append(image)
    # for i in range(len(images)):
    #     plt.imshow(images[i])
    images = numpy.asarray(images)
    images = images.swapaxes(2, 3).swapaxes(1, 2)
    train_test_split = int(round(len(images) * 0.9))
    train_images = images[0:train_test_split, :, :, :]
    test_images = images[train_test_split:, :, :, :]
    return train_images, test_images


def load_mnist(file_name='../mnist.h5'):
    script_dir = os.path.dirname(__file__)  # absolute dir the script is in
    f = h5py.File(os.path.join(script_dir, file_name))
    train_images = f['train'].value.reshape(-1, 28, 28)
    train_images = numpy.expand_dims(train_images, 1)
    test_images = f['test'].value.reshape(-1, 28, 28)
    test_images = numpy.expand_dims(test_images, 1)
    return train_images, test_images


def generate_images(args, images, m_dict, reverse_m_dict):
    im_size, m_range, batch_size = args.image_size, args.motion_range, args.batch_size
    im_channel = images.shape[1]
    idx = numpy.random.permutation(images.shape[0])
    im1 = images[idx[0:batch_size], :, :, :]
    m_label = numpy.random.randint(0, len(m_dict), size=batch_size)
    m_x = numpy.zeros(batch_size).astype(int)
    m_y = numpy.zeros(batch_size).astype(int)
    for i in range(batch_size):
        (m_x[i], m_y[i]) = reverse_m_dict[m_label[i]]
    im2 = move_image(im1, m_x, m_y)
    return im1, im2, m_label


def move_image(im, m_x, m_y):
    [batch_size, im_channel, _, im_size] = im.shape
    m_range_x = numpy.max(numpy.abs(m_x).reshape(-1))
    m_range_y = numpy.max(numpy.abs(m_y).reshape(-1))
    m_range = max(m_range_x, m_range_y).astype(int)
    # im_big = numpy.random.rand(batch_size, im_channel, im_size + m_range * 2, im_size + m_range * 2)
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
    m_dict, reverse_m_dict = motion_dict(1)
    args = learning_args.parse_args()
    train_images, test_images = load_mnist()
    generate_images(args, train_images, m_dict, reverse_m_dict)

if __name__ == '__main__':
    unit_test()
