import tensorflow as tf
from .resnet_layer import ResnetBlock
from tensorflow.keras.layers import Conv2D, LeakyReLU


class SmallNetEncode(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=(3, 3), strides=1, padding="same", activation=None):
        super(SmallNetEncode, self).__init__()
        self.conv = Conv2D(channels,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           activation=activation)
        self.activation = LeakyReLU()

    def __call__(self, inputs, **kwargs):
        x = self.conv(inputs)
        return self.activation(x)


class SmallNetDecode(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size=(3, 3), strides=1, padding="same", activation=None, use_reshape=False):
        super(SmallNetDecode, self).__init__()
        self.use_reshape = use_reshape
        self.conv = Conv2D(channels,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding,
                           activation=activation)
        self.activation = LeakyReLU()

    def __call__(self, inputs_1, inputs_2=None):
        if self.use_reshape:
            h, w = tf.shape(inputs_1)[1], tf.shape(inputs_1)[2]
            inputs_1 = tf.image.resize(inputs_1, (h * 2, w * 2))
        if inputs_2 is not None:
            x = self.conv(inputs_1 + inputs_2)
        else:
            x = self.conv(inputs_1)
        return self.activation(x)


class Unet(tf.keras.Model):
    def __init__(self, channels=32, num_blocks=4, use_clip=False):
        super(Unet, self).__init__()
        self.num_blocks = num_blocks
        self.use_clip = use_clip
        self.resnet_block = ResnetBlock(channels * 4)

        # Encode
        # Block_1:
        self.block_1 = SmallNetEncode(channels, kernel_size=(7, 7))

        # Block_2:
        self.block_2 = SmallNetEncode(channels, strides=2)
        self.block_2_1 = SmallNetEncode(channels * 2)

        # Block_3:
        self.block_3 = SmallNetEncode(channels * 2, strides=2)
        self.block_3_1 = SmallNetEncode(channels * 4)

        # Block_4:
        self.block_4 = SmallNetEncode(channels * 2)

        # Decode:
        # Block_5:
        self.block_5 = SmallNetDecode(channels * 2, use_reshape=True)
        self.block_5_1 = SmallNetDecode(channels)

        # Block_6:
        self.block_6 = SmallNetDecode(channels, use_reshape=True)
        self.block_6_1 = Conv2D(3, padding="same", kernel_size=(7, 7))

    def __call__(self, inputs, **kwargs):
        # Block 1:
        x_1 = self.block_1(inputs)

        # Block 2:
        x_2 = self.block_2_1(self.block_2(x_1))

        # Block 3:
        x_3 = self.block_3_1(self.block_3(x_2))

        for _ in range(self.num_blocks):
            x_3 = self.resnet_block(x_3)

        # Block 4:
        x_3 = self.block_4(x_3)

        # Block 5:
        x_4 = self.block_5_1(self.block_5(x_3, x_2))

        # Block 6:
        x_4 = self.block_6_1(self.block_6(x_4, x_1))

        if self.use_clip:
            x_4 = tf.clip_by_value(x_4, -1, 1)

        return x_4


if __name__ == '__main__':
    samples = tf.random.uniform(shape=(10, 256, 256, 3), minval=0., maxval=1.)
    gen = Unet()
    assert gen(samples).shape == samples.shape
