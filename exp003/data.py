import numpy
import learning_args
import matplotlib.pyplot as plt
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


def generate_images(args, m_dict, reverse_m_dict):
    im_size, m_range, batch_size = args.image_size, args.motion_range, args.batch_size
    im1 = numpy.random.rand(batch_size, 1, im_size, im_size)
    m_label = numpy.random.randint(0, len(m_dict), size=batch_size)
    gt_motion = numpy.zeros((batch_size, 1, im_size, im_size))
    m_x = numpy.zeros(batch_size).astype(int)
    m_y = numpy.zeros(batch_size).astype(int)
    for i in range(batch_size):
        (m_x[i], m_y[i]) = reverse_m_dict[m_label[i]]
        gt_motion[i, :, :, :] = m_label[i]
    im2 = move_image(im1, m_x, m_y)
    im3 = move_image(im2, m_x, m_y)
    if False:
        display(im1, im2, im3, gt_motion)
    return im1, im2, im3, gt_motion.astype(int)


def move_image(im, m_x, m_y):
    [batch_size, im_channel, _, im_size] = im.shape
    m_range_x = numpy.max(numpy.abs(m_x).reshape(-1))
    m_range_y = numpy.max(numpy.abs(m_y).reshape(-1))
    m_range = max(m_range_x, m_range_y).astype(int)
    im_big = numpy.random.rand(batch_size, im_channel, im_size + m_range * 2, im_size + m_range * 2)
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
    generate_images(args, m_dict, reverse_m_dict)

if __name__ == '__main__':
    unit_test()
