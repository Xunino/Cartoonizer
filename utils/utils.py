import os
import cv2
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from skimage.color import label2rgb
from skimage.segmentation import slic
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def simple_superpixel(batch_image, seg_num=200, sigma=1.2):
    def process_slic(image):
        seg_label = slic(image, n_segments=seg_num, sigma=sigma,
                         compactness=10, convert2lab=True)
        image = label2rgb(seg_label, image, kind='avg')
        return image

    num_job = np.shape(batch_image)[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(process_slic)(image) for image in batch_image)
    return np.array(batch_out)


def color_shift(image1, image2, mode='uniform'):
    b1, g1, r1 = tf.split(image1, num_or_size_splits=3, axis=3)
    b2, g2, r2 = tf.split(image2, num_or_size_splits=3, axis=3)
    if mode == 'uniform':
        b_weight = tf.random.uniform(shape=[1], minval=0.014, maxval=0.214)
        g_weight = tf.random.uniform(shape=[1], minval=0.487, maxval=0.687)
        r_weight = tf.random.uniform(shape=[1], minval=0.199, maxval=0.399)
    else:  # norm
        b_weight = tf.random.normal(shape=[1], mean=0.114, stddev=0.1)
        g_weight = tf.random.normal(shape=[1], mean=0.587, stddev=0.1)
        r_weight = tf.random.normal(shape=[1], mean=0.299, stddev=0.1)

    output1 = (b_weight * b1 + g_weight * g1 + r_weight * r1) / (b_weight + g_weight + r_weight)
    output2 = (b_weight * b2 + g_weight * g2 + r_weight * r2) / (b_weight + g_weight + r_weight)
    return output1, output2


if __name__ == '__main__':
    sample_1 = tf.random.uniform(shape=(1, 256, 256, 3), maxval=1.)
    sample_2 = tf.random.uniform(shape=(1, 256, 256, 3), maxval=1.)
    out_1, out_2 = simple_superpixel(sample_1)
    print(out_1.shape)
