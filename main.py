import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from moduls.unet_model import Unet
from moduls.discriminator_spectral_norm import DiscriminatorSN
from utils.guided_fillter import guided_filter
from dataloader import DataLoader
from losses import *
from utils.utils import *


class Trainer:
    def __init__(self,
                 photo_path,
                 cartoon_path,
                 epochs=100,
                 image_shape=256,
                 learning_rate=2e-4,
                 channels=256,
                 batch_size=32):
        self.photo_path = photo_path
        self.cartoon_path = cartoon_path
        self.epochs = epochs
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.channels = channels

        # GAN Model
        self.generator = Unet(channels)

        # Discriminator Model
        self.disc_sn = DiscriminatorSN(self.channels)

        # GAN Optimizer
        self.g_optimizer = Adam(learning_rate, beta_1=0.5, beta_2=0.99)

        # Discriminator Optimizer
        self.d_optimizer = Adam(learning_rate, beta_1=0.5, beta_2=0.99)

    def train_step(self):
        input_photo = DataLoader(self.photo_path, image_shape=self.image_shape, batch_size=self.batch_size).run()
        input_cartoon = DataLoader(self.cartoon_path, image_shape=self.image_shape, batch_size=self.batch_size).run()

        # input_photo = DataLoader(self.photo_path, image_shape=self.image_shape, batch_size=self.batch_size).run()
        # input_cartoon = DataLoader(self.cartoon_path, image_shape=self.image_shape, batch_size=self.batch_size).run()
        for epoch in range(self.epochs):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # Gen
                output = self.generator(input_photo)
                output = guided_filter(input_photo, output, r=1)

                blur_fake = guided_filter(output, output, r=5, eps=2e-1)
                blur_cartoon = guided_filter(input_cartoon, input_cartoon, r=5, eps=2e-1)

                gray_fake, gray_cartoon = color_shift(output, input_cartoon)

                d_loss_gray, g_loss_gray = lsgan_loss(self.disc_sn,
                                                      gray_cartoon, gray_fake,
                                                      patch=True)
                d_loss_blur, g_loss_blur = lsgan_loss(self.disc_sn,
                                                      blur_cartoon, blur_fake,
                                                      patch=True)

                vgg_photo = vgg19(input_photo)
                vgg_output = vgg19(output)
                superpixel_out = simple_superpixel(output)
                vgg_superpixel = vgg19(superpixel_out)

                photo_loss = content_loss(vgg_photo, vgg_output)
                superpixel_loss = content_loss(vgg_superpixel, vgg_photo)

                recon_loss = photo_loss + superpixel_loss
                tv_loss = total_variation_loss(output)

                g_loss_total = 1e4 * tv_loss + 1e-1 * g_loss_blur + g_loss_gray + 2e2 * recon_loss
                d_loss_total = d_loss_blur + d_loss_gray

                print("epoch {} -- g_loss_total: {} -- d_loss_total: {}".format(epoch, g_loss_total.numpy(),
                                                                                d_loss_total.numpy()))
            # Generator
            gradients_of_generator = gen_tape.gradient(g_loss_total, self.generator.trainable_variables)
            self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

            # Discriminator
            gradients_of_discriminator = disc_tape.gradient(d_loss_total, self.disc_sn.trainable_variables)
            self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.disc_sn.trainable_variables))

    def test_step(self):
        pass


if __name__ == '__main__':
    real_samples = "dataset/photos"
    cartoon_samples = "dataset/cartoons"
    train = Trainer(real_samples, cartoon_samples, image_shape=256, epochs=5, batch_size=1)

    train.train_step()
