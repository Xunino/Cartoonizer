import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU


class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels=32):
        super(ResnetBlock, self).__init__()
        self.conv_1 = Conv2D(out_channels, kernel_size=(3, 3), padding="same", activation=None)
        self.conv_2 = Conv2D(out_channels, kernel_size=(3, 3), padding="same", activation=None)
        self.activation = LeakyReLU()

    def __call__(self, inputs, *args, **kwargs):
        x = self.conv_1(inputs)
        x = self.activation(x)
        x = self.conv_2(x)
        return x + inputs
