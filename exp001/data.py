import numpy


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


def generate_images(args, m_dict, reverse_m_dict):
    im_size, m_range, batch_size = args.image_size, args.motion_range, args.batch_size
    im1 = numpy.random.rand(batch_size, 1, im_size, im_size)
    im_big = numpy.random.rand(batch_size, 1, im_size+m_range*2, im_size+m_range*2)
    im_big[:, :, m_range:-m_range, m_range:-m_range] = im1
    im2 = numpy.zeros((batch_size, 1, im_size, im_size))
    m_label = numpy.random.randint(0, len(m_dict), size=batch_size)
    for i in range(batch_size):
        (m_x, m_y) = reverse_m_dict[m_label[i]]
        im2[i, :, :, :] = im_big[i, :, m_range+m_y:m_range+m_y+im_size, m_range+m_x:m_range+m_x+im_size]
    return im1, im2, m_label

