import os
from PIL import Image
import numpy
import matplotlib.pyplot as plt


def get_diff_idx(robot_dir='robot-64'):
    diff_file = open('./robot_diff', 'w')
    for i in range(1, 1219):
        for j in range(25):
            print('sequence %d, %d' % (i, j))
            image_name = os.path.join(robot_dir, str(i), str(j), str(0) + '.jpg')
            im_old = numpy.array(Image.open(image_name)) / 255.0
            im_old = im_old[:, :64, :]
            im_diff_max = []
            for k in range(1, 25, 2):
                image_name = os.path.join(robot_dir, str(i), str(j), str(k) + '.jpg')
                im = numpy.array(Image.open(image_name)) / 255.0
                im = im[:, :64, :]
                im_diff = numpy.abs(im - im_old)
                if False:
                    display(im_old, im, im_diff)
                im_diff_max.append("{:.2f}".format(im_diff.max()))
                im_diff = im_diff.sum()
                diff_file.write('%d\t%d\t%d\t%.2f\n' % (i, j, k, im_diff))
                im_old = im
            print(" ".join(im_diff_max))


def display(im_old, im, im_diff):
    img_size = 64
    width, height = get_img_size(1, 3, img_size)
    img = numpy.ones((height, width, 3))

    x1, y1, x2, y2 = get_img_coordinate(1, 1, img_size)
    img[y1:y2, x1:x2, :] = im_old

    x1, y1, x2, y2 = get_img_coordinate(1, 2, img_size)
    img[y1:y2, x1:x2, :] = im

    x1, y1, x2, y2 = get_img_coordinate(1, 3, img_size)
    img[y1:y2, x1:x2, :] = im_diff

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

