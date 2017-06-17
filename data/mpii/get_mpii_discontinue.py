import os
from PIL import Image
import numpy
import pickle
import matplotlib.pyplot as plt


def get_diff_idx(meta_file='mpii_meta_2.pkl'):
    meta = pickle.load(open(meta_file))
    for k, v in meta.iteritems():
        # print v[1]
        im_diff_sum = v[3][1:]
        # print im_diff_sum
        outlier = mad_based_outlier(numpy.asarray(im_diff_sum), 10)
        outlier_file = []
        for i in range(len(outlier)):
            if outlier[i]:
                outlier_file.append(v[2][i+1])
        if len(outlier_file) > 0:
            print v[1] + " " + " ".join(outlier_file)
            im_diff_str = ["{:.0f}".format(im_diff) for im_diff in im_diff_sum]
            # print " ".join(im_diff_str)


def mad_based_outlier(points, thresh=3.5):
    """
    median absolute deviation (MAD) test for outlier
    https://stackoverflow.com/questions/22354094/pythonic-way-of-detecting-outliers-in-one-dimensional-observation-data
    """
    median = numpy.median(points)
    diff = (points - median)**2
    diff = numpy.sqrt(diff)
    med_abs_deviation = numpy.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

if __name__ == '__main__':
    get_diff_idx()
