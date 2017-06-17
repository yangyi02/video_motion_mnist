import os
from PIL import Image
import numpy
import pickle
import matplotlib.pyplot as plt


def get_diff_idx(input_dir='mpii-64', meta_file='mpii_meta.pkl', output_meta_file='mpii_meta_2.pkl'):
    diff_file = open('./mpii_diff', 'w')
    meta = pickle.load(open(meta_file))
    for k, v in meta.iteritems():
        image_dir = v[0]
        sub_dir = v[1]
        file_names = v[2]
        print('sequence %s, %s' % (image_dir, sub_dir))
        image_name = os.path.join(input_dir, image_dir, sub_dir, file_names[0])
        im_old = numpy.array(Image.open(image_name)) / 255.0
        im_diff_sum = [0]
        im_diff_str = []
        for i in range(1, len(file_names)):
            image_name = os.path.join(input_dir, image_dir, sub_dir, file_names[i])
            im = numpy.array(Image.open(image_name)) / 255.0
            im_diff = numpy.abs(im - im_old)
            if False:
                display(im_old, im, im_diff)
            im_diff = im_diff.sum()
            im_diff_sum.append(im_diff)
            im_diff_str.append("{:.2f}".format(im_diff))
            diff_file.write('%s\t%s\t%s\t%.2f\n' % (image_dir, sub_dir, file_names[i], im_diff))
            im_old = im
        print(" ".join(im_diff_str))
        meta[k].append(im_diff_sum)
    pickle.dump(meta, open(output_meta_file, 'w'))


def display(im_old, im, im_diff):
    img_size = 64
    width, height = get_img_size(1, 3, img_size)
    img = numpy.ones((height, width, 3))

    x1, y1, x2, y2 = get_img_coordinate(1, 1, img_size)
    img[y1:y2, x1:x2, :] = im_old[:64, :64, :]

    x1, y1, x2, y2 = get_img_coordinate(1, 2, img_size)
    img[y1:y2, x1:x2, :] = im[:64, :64, :]

    x1, y1, x2, y2 = get_img_coordinate(1, 3, img_size)
    img[y1:y2, x1:x2, :] = im_diff[:64, :64, :]

    # print('image diff: %.2f' % im_diff.sum())

    plt.figure(1)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def get_img_size(n_row, n_col, img_size):
    height = n_row * img_size + (n_row - 1) * int(img_size/10)
    width = n_col * img_size + (n_col - 1) * int(img_size/10)
    return width, height


def get_img_coordinate(row, col, img_size):
    y1 = (row - 1) * img_size + (row - 1) * int(img_size/10)
    y2 = y1 + img_size
    x1 = (col - 1) * img_size + (col - 1) * int(img_size/10)
    x2 = x1 + img_size
    return x1, y1, x2, y2


def main():
    get_diff_idx()

if __name__ == '__main__':
    main()

