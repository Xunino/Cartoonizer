import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19

VGG_MEAN = [103.939, 116.779, 123.68]


class VGG19Content:
    def __init__(self):
        pass

    def __call__(self, x, *args, **kwargs):
        h, w, c = tf.shape(x)[1:4]
        shape = (h, w, c)
        rgb_scaled = (x + 1) * 127.5
        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        x = tf.concat(axis=3, values=[blue - VGG_MEAN[0],
                                      green - VGG_MEAN[1], red - VGG_MEAN[2]])

        back_bone = VGG19(include_top=False, weights="imagenet", input_shape=shape)
        for layer in back_bone.layers:
            layer.trainable = False

        model = Model(inputs=back_bone.input, outputs=back_bone.get_layer("block4_pool").output)

        return model(x)


def content_loss(real_photo, fake_photo):
    h, w, c = tf.shape(real_photo)[1:]
    return tf.reduce_mean(tf.compat.v1.losses.absolute_difference(real_photo, fake_photo)) / float(h * w * c)


def total_variation_loss(image, k_size=1):
    h, w = image.get_shape().as_list()[1:3]
    tv_h = tf.reduce_mean((image[:, k_size:, :, :] - image[:, :h - k_size, :, :]) ** 2)
    tv_w = tf.reduce_mean((image[:, :, k_size:, :] - image[:, :, :w - k_size, :]) ** 2)
    tv_loss = (tv_h + tv_w) / (3 * h * w)
    return tv_loss


def lsgan_loss(discriminator, real, fake, patch=True):
    real_logit = discriminator(real, patch=patch)
    fake_logit = discriminator(fake, patch=patch)

    g_loss = tf.reduce_mean((fake_logit - 1) ** 2)
    d_loss = 0.5 * (tf.reduce_mean((real_logit - 1) ** 2) + tf.reduce_mean(fake_logit ** 2))

    return d_loss, g_loss
