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
    im_idx = range(0, 11, 2)
    n_image = len(im_idx)
    images = numpy.zeros((batch_size, height, width, im_channel * n_image))
    idx = numpy.random.permutation(len(robot_meta))[0:batch_size]
    for i in range(batch_size):
        [dir_id, sub_dir_id] = robot_meta[idx[i]]
        for j in range(n_image):
            image_name = os.path.join(robot_dir, str(dir_id), str(sub_dir_id), str(im_idx[j]) + '.jpg')
            im = numpy.array(Image.open(image_name)) / 255.0
            images[i, :, :, j*im_channel:(j+1)*im_channel] = im[:, -64:, :]
    images = images.transpose((0, 3, 1, 2))
    return images


def generate_batch(args, images):
    batch_size, height, width, im_channel = args.batch_size, 64, 64, 3
    n_image = images.shape[1] / im_channel
    n = 5
    idx = numpy.random.randint(0, n_image - n, size=batch_size)
    im_input = numpy.zeros((batch_size, im_channel*n, height, width))
    im_output = numpy.zeros((batch_size, im_channel, height, width))
    for i in range(batch_size):
        im_input[i, :, :, :] = images[i, idx[i] * im_channel:(idx[i] + n) * im_channel, :, :]
        im_output[i, :, :, :] = images[i, (idx[i] + n) * im_channel:(idx[i] + n + 1) * im_channel, :, :]
    return im_input, im_output


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
    im_input, im_output = generate_batch(args, images)
    if True:
        im1 = im_input[:, 0:3, :, :]
        im2 = im_input[:, 3:6, :, :]
        im3 = im_output
        display(im1, im2, im3)

if __name__ == '__main__':
    unit_test()
