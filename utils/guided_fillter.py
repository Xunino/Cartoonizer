import tensorflow as tf
import numpy as np


def tf_box_filter(x, r):
    ch = tf.shape(x)[-1]
    weight = 1 / ((2 * r + 1) ** 2)  # r=1 -> weight = 1/9
    box_kernel = weight * tf.ones((2 * r + 1, 2 * r + 1, ch, 1), dtype=tf.float32)  # r=1 -> box_kernel: shape[3, 3, 3, 1]
    output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')
    return output


def guided_filter(x, y, r, eps=1e-2):
    x_shape = tf.shape(x)

    N = tf_box_filter(tf.ones((1, x_shape[1], x_shape[2], 1), dtype=x.dtype), r)

    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x = tf_box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N

    output = mean_A * x + mean_b

    return output
