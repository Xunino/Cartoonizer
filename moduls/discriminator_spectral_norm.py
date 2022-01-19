import tensorflow as tf
from .spectral_norm_layer import conv_spectral_norm
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization
from tensorflow.keras.models import Model


class DiscriminatorBN(tf.keras.layers.Layer):
    def __init__(self, channels, padding="same"):
        super(DiscriminatorBN, self).__init__()
        self.channels = channels
        self.padding = padding

    def __call__(self, x, patch=True):
        for idx in range(3):
            x = Conv2D(filters=self.channels * 2 ** idx, kernel_size=(3, 3), strides=2,
                       padding=self.padding,
                       name="conv_{}_1".format(idx))(x)
            x = BatchNormalization()(x)
            x = tf.nn.leaky_relu(x)

            x = Conv2D(filters=self.channels * 2 ** idx, kernel_size=(3, 3), strides=1,
                       padding=self.padding,
                       name="conv_{}_1".format(idx))(x)
            x = BatchNormalization()(x)
            x = tf.nn.leaky_relu(x)

        if patch:
            x = Conv2D(1, kernel_size=(3, 3), padding=self.padding, name="conv_out")(x)

        else:
            x = tf.reduce_mean(x, axis=[1, 2])
            x = Dense(1)(x)
        return x


class DiscriminatorSN(tf.keras.layers.Layer):
    def __init__(self, channels=32):
        super(DiscriminatorSN, self).__init__()
        self.channels = channels
        self.dense = Dense(1)

    def __call__(self, x, patch=True):
        for idx in range(3):
            x = conv_spectral_norm(x, channel=self.channels * 2 ** idx, k_size=[3, 3],
                                   stride=2, name="conv{}_1".format(idx))
            x = tf.nn.leaky_relu(x)

            x = conv_spectral_norm(x, channel=self.channels * 2 ** idx, k_size=[3, 3],
                                   stride=1, name="conv{}_2".format(idx))
            x = tf.nn.leaky_relu(x)

        if patch:
            x = conv_spectral_norm(x, 1, [1, 1], name="conv_out")

        else:
            x = tf.reduce_mean(x, axis=[1, 2])
            x = self.dense(x)

        return x


if __name__ == '__main__':
    sample = tf.random.uniform(shape=[1, 256, 256, 3], minval=0., maxval=1.)
    disc_sn = DiscriminatorBN(sample)
    print(disc_sn.shape)
