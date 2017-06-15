import os
import numpy
import matplotlib.pyplot as plt
from skimage import io, transform
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


def get_robot_meta():
    robot_dir = '../robot-64'
    robot_meta = {}
    cnt = 0
    for i in range(1, 1219):
        for j in range(25):
            robot_meta[cnt] = [i, j]
            cnt += 1
    return robot_meta, robot_dir


def load_robot_data(args, robot_meta, robot_dir):
    batch_size, height, width, im_channel = args.batch_size, 64, 64, 3
    images = numpy.zeros((batch_size, height, width, im_channel * 25))
    idx = numpy.random.permutation(len(robot_meta))[0:batch_size]
    for i in range(batch_size):
        [dir_id, sub_dir_id] = robot_meta[idx[i]]
        for j in range(25):
            image_name = os.path.join(robot_dir, str(dir_id), str(sub_dir_id), str(j) + '.jpg')
            im = numpy.array(Image.open(image_name)) / 255.0
            images[i, :, :, j*im_channel:(j+1)*im_channel] = im[:, :64, :]
    images = images.transpose((0, 3, 1, 2))
    return images


def generate_batch(args, images):
    batch_size, height, width, im_channel = args.batch_size, 64, 64, 3
    idx = numpy.random.randint(1, 24, size=batch_size)
    im1 = numpy.zeros((batch_size, im_channel, height, width))
    im2 = numpy.zeros((batch_size, im_channel, height, width))
    im3 = numpy.zeros((batch_size, im_channel, height, width))
    for i in range(batch_size):
        im1[i, :, :, :] = images[i, (idx[i] - 1) * im_channel:idx[i] * im_channel, :, :]
        im2[i, :, :, :] = images[i, idx[i] * im_channel:(idx[i] + 1) * im_channel, :, :]
        im3[i, :, :, :] = images[i, (idx[i] + 1) * im_channel:(idx[i] + 2) * im_channel, :, :]
    return im1, im2, im3


def display(im1, im2, im3):
    plt.figure(1)
    plt.subplot(1, 3, 1)
    plt.imshow(im1[0, :, :, :].squeeze().transpose(1, 2, 0))
    plt.subplot(1, 3, 2)
    plt.imshow(im2[0, :, :, :].squeeze().transpose(1, 2, 0))
    plt.subplot(1, 3, 3)
    plt.imshow(im3[0, :, :, :].squeeze().transpose(1, 2, 0))
    plt.show()


def unit_test():
    m_dict, reverse_m_dict, m_kernel = motion_dict(1)
    args = learning_args.parse_args()
    robot_meta, robot_dir = get_robot_meta()
    images = load_robot_data(args, robot_meta, robot_dir)
    [_, _, args.image_height, args.image_width] = images.shape
    im1, im2, im3 = generate_batch(args, images)
    if True:
        display(im1, im2, im3)

if __name__ == '__main__':
    unit_test()
