import tensorflow as tf
from .spectral_norm_layer import conv_spectral_norm
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Dense


class DiscriminatorSN(tf.keras.layers.Layer):
    def __init__(self, channels=32, ):
        super(DiscriminatorSN, self).__init__()
        self.channels = channels

        self.bn_1 = BatchNormalization(center=True, scale=True)
        self.bn_2 = BatchNormalization(center=True, scale=True)

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


if __name__ == '__main__':
    sample = tf.random.uniform(shape=[1, 256, 256, 3], minval=0., maxval=1.)
    disc_sn = DiscriminatorSN()
    print(disc_sn(sample).shape)
