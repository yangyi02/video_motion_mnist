import os
import numpy
from skimage import io, transform
import matplotlib.pyplot as plt
import h5py


def motion_dict(m_range):
    m_dict, reverse_m_dict = {}, {}
    x = numpy.linspace(-m_range, m_range, 2*m_range+1)
    y = numpy.linspace(-m_range, m_range, 2*m_range+1)
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
    # im_big = numpy.random.rand(batch_size, im_channel, im_size+m_range*2, im_size+m_range*2)
    im_big = numpy.zeros((batch_size, im_channel, im_size + m_range * 2, im_size + m_range * 2))
    im_big[:, :, m_range:-m_range, m_range:-m_range] = im1
    im2 = numpy.zeros((batch_size, im_channel, im_size, im_size))
    m_label = numpy.random.randint(0, len(m_dict), size=batch_size)
    for i in range(batch_size):
        (m_x, m_y) = reverse_m_dict[m_label[i]]
        im2[i, :, :, :] = im_big[i, :, m_range+m_y:m_range+m_y+im_size, m_range+m_x:m_range+m_x+im_size]
    return im1, im2, m_label