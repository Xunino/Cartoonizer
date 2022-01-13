import tensorflow as tf
from .spectral_norm_layer import conv_spectral_norm
from tensorflow.keras.layers import LeakyReLU, Dense, BatchNormalization, Conv2D
from tensorflow.keras.models import Model


class DiscriminatorSN(Model):
    def __init__(self, channels=32):
        super(DiscriminatorSN, self).__init__()
        self.channels = channels

        self.activate_1 = LeakyReLU()
        self.activate_2 = LeakyReLU()

        self.dense = Dense(1)

    def __call__(self, x, patch=True):
        for idx in range(3):
            x = conv_spectral_norm(x, channel=self.channels * 2 ** idx, k_size=[3, 3],
                                   stride=2, name="conv{}_1".format(idx))
            x = self.activate_1(x)

            x = conv_spectral_norm(x, channel=self.channels * 2 ** idx, k_size=[3, 3],
                                   stride=1, name="conv{}_2".format(idx))
            x = self.activate_2(x)

        if patch:
            x = conv_spectral_norm(x, 1, [1, 1], name="conv_out")

        else:
            x = tf.reduce_mean(x, axis=[1, 2])
            x = self.dense(x)

        return x


class DiscriminatorBN(Model):
    def __init__(self, channels=32):
        super(DiscriminatorBN, self).__init__()
        self.channels = channels

        self.bn_1 = BatchNormalization()
        self.bn_2 = BatchNormalization()

        self.activate_1 = LeakyReLU()
        self.activate_2 = LeakyReLU()

        self.dense = Dense(1)

    def __call__(self, x, patch=True):
        for idx in range(3):
            x = tf.nn.conv2d(x, filtersself.channels * 2 ** idx, k_size=[3, 3],
                             stride=2, name="conv{}_1".format(idx))
            x = self.activate_1(x)

            x = conv_spectral_norm(x, channel=self.channels * 2 ** idx, k_size=[3, 3],
                                   stride=1, name="conv{}_2".format(idx))
            x = self.activate_2(x)

        if patch:
            x = conv_spectral_norm(x, 1, [1, 1], name="conv_out")

        else:
            x = tf.reduce_mean(x, axis=[1, 2])
            x = self.dense(x)

        return x


def disc_bn(x, scale=1, channel=32, is_training=True,
            name='discriminator', patch=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse):

        for idx in range(3):
            x = slim.convolution2d(x, channel * 2 ** idx, [3, 3], stride=2, activation_fn=None)
            x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
            x = tf.nn.leaky_relu(x)

            x = slim.convolution2d(x, channel * 2 ** idx, [3, 3], activation_fn=None)
            x = slim.batch_norm(x, is_training=is_training, center=True, scale=True)
            x = tf.nn.leaky_relu(x)

        if patch:
            x = slim.convolution2d(x, 1, [1, 1], activation_fn=None)
        else:
            x = tf.reduce_mean(x, axis=[1, 2])
            x = slim.fully_connected(x, 1, activation_fn=None)

        return x


if __name__ == '__main__':
    sample = tf.random.uniform(shape=[1, 256, 256, 3], minval=0., maxval=1.)
    disc_sn = DiscriminatorSN()
    print(disc_sn(sample).shape)
