import os
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from moduls.unet_model import Unet
from moduls.discriminator_spectral_norm import DiscriminatorSN
from utils.guided_fillter import guided_filter
from dataloader import DataLoader
from losses import VGG19Content, lsgan_loss, content_loss, total_variation_loss
from utils.utils import write_batch_image, color_shift, simple_superpixel

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HOME = os.getcwd()


class Trainer:
    def __init__(self,
                 photo_face_path,
                 cartoon_face_path,
                 photo_scenery_path,
                 cartoon_scenery_path,
                 saved_weights="weights",
                 epochs=100,
                 image_shape=256,
                 learning_rate=2e-4,
                 channels=32,
                 batch_size=32,
                 use_parallel=True):
        self.photo_face_path = photo_face_path
        self.cartoon_face_path = cartoon_face_path

        self.photo_scenery_path = photo_scenery_path
        self.cartoon_scenery_path = cartoon_scenery_path

        self.epochs = epochs
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.channels = channels
        self.use_parallel = use_parallel

        self.loss_min = 2.0

        self.vgg = VGG19Content()

        # GAN Model
        self.generator = Unet(channels)

        # Discriminator Model
        self.disc_sn = DiscriminatorSN(self.channels)

        # GAN Optimizer
        self.g_optimizer = Adam(learning_rate, beta_1=0.5, beta_2=0.99)

        # Discriminator Optimizer
        self.d_optimizer = Adam(learning_rate, beta_1=0.5, beta_2=0.99)

        # Checkpoint
        self.saved_gen_weights = os.path.join(HOME, saved_weights, "generator/")
        self.saved_disc_weights = os.path.join(HOME, saved_weights, "discriminator/")
        os.makedirs(self.saved_gen_weights, exist_ok=True)
        os.makedirs(self.saved_disc_weights, exist_ok=True)

        ckpt_gen = tf.train.Checkpoint(generator=self.generator,
                                       optimizer=self.g_optimizer)
        ckpt_disc = tf.train.Checkpoint(generator=self.disc_sn,
                                        optimizer=self.d_optimizer)

        self.ckpt_gen_manager = tf.train.CheckpointManager(ckpt_gen, self.saved_gen_weights, max_to_keep=2)
        self.ckpt_disc_manager = tf.train.CheckpointManager(ckpt_disc, self.saved_disc_weights, max_to_keep=2)

    def train_step(self):

        # Start training
        for epoch in range(self.epochs + 1):
            g_loss = 0
            d_loss = 0

            # Photo face and cartoon face
            input_photo_face, len_photo_face = DataLoader(self.photo_face_path, image_shape=self.image_shape,
                                                          batch_size=self.batch_size).run()
            input_cartoon_face, len_cartoon_face = DataLoader(self.cartoon_face_path, image_shape=self.image_shape,
                                                              batch_size=self.batch_size).run()

            # Photo scenery and cartoon scenery
            input_photo_scenery, len_photo_scenery = DataLoader(self.photo_scenery_path, image_shape=self.image_shape,
                                                                batch_size=self.batch_size).run()
            input_cartoon_scenery, len_cartoon_scenery = DataLoader(self.cartoon_scenery_path,
                                                                    image_shape=self.image_shape,
                                                                    batch_size=self.batch_size).run()

            min_len = min(len_photo_face, len_cartoon_face, len_photo_scenery,
                          len_cartoon_scenery) // self.batch_size + 1
            pbar = tqdm(range(min_len))
            for iterator in pbar:
                if iterator % 5 == 0:
                    input_photo = input_photo_face
                    input_cartoon = input_cartoon_face
                else:
                    input_photo = input_photo_scenery
                    input_cartoon = input_cartoon_scenery

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

                    vgg_photo = self.vgg(input_photo)
                    vgg_output = self.vgg(output)
                    superpixel_out = simple_superpixel(output, use_parallel=self.use_parallel)
                    vgg_superpixel = self.vgg(superpixel_out)

                    photo_loss = content_loss(vgg_photo, vgg_output)
                    superpixel_loss = content_loss(vgg_superpixel, vgg_photo)

                    recon_loss = photo_loss + superpixel_loss
                    tv_loss = total_variation_loss(output)

                    g_loss += (1e4 * tv_loss + 1e-1 * g_loss_blur + g_loss_gray + 2e2 * recon_loss) / self.batch_size
                    d_loss += (d_loss_blur + d_loss_gray) / self.batch_size

                    # Generator
                    gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
                    self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

                    # Discriminator
                    gradients_of_discriminator = disc_tape.gradient(d_loss, self.disc_sn.trainable_variables)
                    self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.disc_sn.trainable_variables))

                    pbar.set_description("Epoch {} || g_loss_total: {} || d_loss_total: {}".format(epoch,
                                                                                                   g_loss,
                                                                                                   d_loss))

                    if d_loss < self.loss_min:
                        results = self.generator(input_photo)
                        write_batch_image(results, os.path.join(HOME, "test_images"), "_result.jpg", 1)
                        self.ckpt_gen_manager.save()
                        self.ckpt_disc_manager.save()
                        self.loss_min = d_loss

            g_loss_total = g_loss / min_len
            d_loss_total = d_loss / min_len
            print("Epoch {} || g_loss_total: {} || d_loss_total: {}".format(epoch,
                                                                            g_loss_total,
                                                                            d_loss_total))


if __name__ == '__main__':
    pc = False
    if pc:
        real_face = "dataset/faces"
        cartoon_faces = "dataset/cartoon_faces"
        real_scenery = "dataset/scenery"
        cartoon_scenery = "dataset/cartoon_scenery"
        train = Trainer(real_face, cartoon_faces,
                        real_scenery, cartoon_scenery,
                        image_shape=128, epochs=5,
                        batch_size=6, channels=8, use_parallel=True)
    else:
        real_face = "/content/drive/MyDrive/dataset/faces"
        cartoon_faces = "/content/drive/MyDrive/dataset/cartoon_faces"
        real_scenery = "/content/drive/MyDrive/dataset/scenery"
        cartoon_scenery = "/content/drive/MyDrive/dataset/cartoon_scenery"
        train = Trainer(real_face, cartoon_faces,
                        real_scenery, cartoon_scenery,
                        image_shape=128, epochs=5,
                        batch_size=6, channels=8, use_parallel=False)

    train.train_step()
    """
    Epoch 0 || g_loss_total: 2.736996650695801 || d_loss_total: 0.9677947759628296
    """
