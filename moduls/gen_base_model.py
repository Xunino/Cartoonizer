import tensorflow as tf
from resnet_layer import ResnetBlock
from tensorflow.keras.layers import Conv2D, LeakyReLU, Conv2DTranspose


class Generator(tf.keras.Model):
    def __init__(self, channels=32, num_blocks=4, use_clip=False):
        super(Generator, self).__init__()
        self.num_blocks = num_blocks
        self.use_clip = use_clip
        self.resnet_block = ResnetBlock(channels * 4)

        # Encode
        # Block_1
        self.conv_1 = Conv2D(channels, kernel_size=(7, 7), padding="same", activation=None)
        # Block_2
        self.conv_2 = Conv2D(channels * 2, kernel_size=(3, 3), strides=2, padding="same", activation=None)
        self.conv_3 = Conv2D(channels * 2, kernel_size=(3, 3), padding="same", activation=None)
        # Block_3
        self.conv_4 = Conv2D(channels * 4, kernel_size=(3, 3), padding="same", strides=2, activation=None)
        self.conv_5 = Conv2D(channels * 4, kernel_size=(3, 3), padding="same", activation=None)
        # Encode activation
        self.activation_1 = LeakyReLU()
        self.activation_2 = LeakyReLU()
        self.activation_3 = LeakyReLU()

        # Decode
        # Block_4
        self.conv_transpose_1 = Conv2DTranspose(channels * 2, kernel_size=(3, 3), strides=2, padding="same",
                                                activation=None)
        self.conv_6 = Conv2D(channels * 2, kernel_size=(3, 3), padding="same", activation=None)
        # Block_5
        self.conv_transpose_2 = Conv2DTranspose(channels, kernel_size=(3, 3), strides=2, padding="same",
                                                activation=None)
        self.conv_7 = Conv2D(channels, kernel_size=(3, 3), padding="same", activation=None)
        # Block_6
        self.conv_8 = Conv2D(3, kernel_size=(7, 7), padding="same", activation=None)
        # Decode activation
        self.activation_4 = LeakyReLU()
        self.activation_5 = LeakyReLU()

    def call(self, inputs, **kwargs):
        # Encode
        # Block 1:
        x = self.conv_1(inputs)
        x = self.activation_1(x)

        # Block 2:
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.activation_2(x)

        # Block 3:
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.activation_3(x)

        # Resnet block
        for _ in range(self.num_blocks):
            x = self.resnet_block(x)

        # Decode
        # Block 4:
        x = self.conv_transpose_1(x)
        x = self.conv_6(x)
        x = self.activation_4(x)

        # Block 5:
        x = self.conv_transpose_2(x)
        x = self.conv_7(x)
        x = self.activation_5(x)

        # Block 6:
        x = self.conv_8(x)

        # Clip value in range [-0.999999, 0.999999]
        if self.use_clip:
            x = tf.clip_by_value(x, -0.999999, 0.999999)

        return x


if __name__ == '__main__':
    inputs = tf.random.uniform(shape=(1, 256, 256, 3), minval=0, maxval=1)
    gen = Generator()
    assert gen.shape == inputs.shape
