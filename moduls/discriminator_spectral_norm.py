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


class ConvBN(tf.keras.layers.Layer):
    def __init__(self, channels=32, kernel_size=(3, 3), strides=1, padding="same"):
        self.conv = Conv2D(channels, kernel_size=kernel_size, strides=strides, padding=padding)
        self.bn = BatchNormalization()
        self.activate = LeakyReLU()

    def __call__(self, x, *args, **kwargs):
        return self.activate(self.bn(self.conv(x)))


class DiscriminatorBN(Model):
    def __init__(self, channels=32):
        super(DiscriminatorBN, self).__init__()
        self.channels = channels

        self.conv_bn_1_1 = ConvBN(channels, kernel_size=(3, 3), strides=2, padding="same")
        self.conv_bn_1_2 = ConvBN(channels, kernel_size=(3, 3), padding="same")

        self.conv_bn_2_1 = ConvBN(channels * 2, kernel_size=(3, 3), strides=2, padding="same")
        self.conv_bn_2_2 = ConvBN(channels * 2, kernel_size=(3, 3), padding="same")

        self.conv_bn_3_1 = ConvBN(channels * 4, kernel_size=(3, 3), strides=2, padding="same")
        self.conv_bn_3_2 = ConvBN(channels * 4, kernel_size=(3, 3), padding="same")

        self.conv_out = Conv2D(1, kernel_size=(1, 1), padding="same")
        self.dense = Dense(1)

    def __call__(self, x, patch=True, *args, **kwargs):
        x = self.conv_bn_1_1(x)
        x = self.conv_bn_1_2(x)

        x = self.conv_bn_2_1(x)
        x = self.conv_bn_2_2(x)

        x = self.conv_bn_3_1(x)
        x = self.conv_bn_3_2(x)

        if patch:
            x = self.conv_out(x)
        else:
            x = tf.reduce_mean(x, axis=[1, 2])
            x = self.dense(x)

        return x


if __name__ == '__main__':
    sample = tf.random.uniform(shape=[1, 256, 256, 3], minval=0., maxval=1.)
    disc_sn = DiscriminatorBN()
    print(disc_sn(sample).shape)
