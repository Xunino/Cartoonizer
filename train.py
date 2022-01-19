import os
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from moduls.unet_model import Unet
from moduls.discriminator_spectral_norm import DiscriminatorSN
from utils.guided_fillter import guided_filter
from dataloader import DataLoader, DataLoaderWithTF
from losses import VGG19Content, lsgan_loss, content_or_structure_loss, total_variation_loss, INCEPTIONContent
from utils.utils import write_batch_image, color_shift, simple_superpixel, get_list_images, selective_adacolor

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
                 use_enhance=True,
                 use_parallel=True,
                 retrain=False):
        self.photo_face_path = photo_face_path
        self.cartoon_face_path = cartoon_face_path

        self.photo_scenery_path = photo_scenery_path
        self.cartoon_scenery_path = cartoon_scenery_path

        self.epochs = epochs
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.channels = channels
        self.use_parallel = use_parallel
        self.use_enhance = use_enhance

        self.high_level_features = VGG19Content()

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

        if retrain:
            print("[INFO] Start Retrain...")
            print("[INFO] Loading model...")
            self.ckpt_gen_manager.restore_or_initialize()
            self.ckpt_disc_manager.restore_or_initialize()
            print("[INFO] DONE!")

    def train_step(self):

        # Start training
        for epoch in range(self.epochs + 1):
            g_loss_total = 0
            d_loss_total = 0

            # Photo face and cartoon face
            input_photo_face = DataLoaderWithTF(get_list_images(self.photo_face_path),
                                                image_shape=self.image_shape,
                                                batch_size=self.batch_size)
            input_cartoon_face = DataLoaderWithTF(get_list_images(self.cartoon_face_path),
                                                  image_shape=self.image_shape,
                                                  batch_size=self.batch_size)

            # Photo scenery and cartoon scenery
            input_photo_scenery = DataLoaderWithTF(get_list_images(self.photo_scenery_path),
                                                   image_shape=self.image_shape,
                                                   batch_size=self.batch_size)
            input_cartoon_scenery = DataLoaderWithTF(get_list_images(self.cartoon_scenery_path),
                                                     image_shape=self.image_shape,
                                                     batch_size=self.batch_size)

            min_len = max(input_photo_face.__len__(), input_cartoon_face.__len__(), input_photo_scenery.__len__(),
                          input_cartoon_scenery.__len__()) // self.batch_size + 1
            pbar = tqdm(range(1, min_len + 1))
            for iterator in pbar:
                if iterator % 5 == 0:
                    input_photo = input_photo_face()
                    input_cartoon = input_cartoon_face()
                else:
                    input_photo = input_photo_scenery()
                    input_cartoon = input_cartoon_scenery()

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    # Gen
                    output = self.generator(input_photo)
                    output = guided_filter(input_photo, output, r=1)  # Smooth

                    """
                        Surface
                    """
                    # Surface output
                    blur_fake = guided_filter(output, output, r=5, eps=2e-1)

                    # Surface cartoon
                    blur_cartoon = guided_filter(input_cartoon, input_cartoon, r=5, eps=2e-1)

                    # Loss surface using disc model
                    d_loss_blur, g_loss_blur = lsgan_loss(self.disc_sn,
                                                          blur_cartoon, blur_fake,
                                                          patch=True)
                    """
                        Texture loss
                    """
                    # Texture output and cartoon
                    gray_fake, gray_cartoon = color_shift(output, input_cartoon)

                    # Loss texture using disc model
                    d_loss_gray, g_loss_gray = lsgan_loss(self.disc_sn,
                                                          gray_cartoon, gray_fake,
                                                          patch=True)
                    """
                        Total variation loss
                    """
                    # Total variation loss
                    tv_loss = total_variation_loss(output)

                    # input images
                    vgg_photo = self.high_level_features(input_photo)
                    # fake images
                    vgg_output = self.high_level_features(output)
                    # Superpixel images
                    if self.use_enhance:
                        superpixel_out = selective_adacolor(output, seg_num=200, use_parallel=self.use_parallel,
                                                            num_job=4)
                    else:
                        superpixel_out = simple_superpixel(output, use_parallel=self.use_parallel, num_job=4)

                    vgg_superpixel = self.high_level_features(superpixel_out)

                    """
                        Recon_loss = content_loss + structure_loss 
                    """
                    # Content loss
                    # Real vs Fake
                    photo_loss = content_or_structure_loss(vgg_photo, vgg_output)

                    # Structure loss
                    # Fake vs Superpixel
                    superpixel_loss = content_or_structure_loss(vgg_superpixel, vgg_photo)
                    recon_loss = photo_loss + superpixel_loss

                    """
                        - gen_loss
                        - disc_loss
                    """
                    g_loss = 1e4 * tv_loss + 1e-1 * g_loss_blur + g_loss_gray + 2e2 * recon_loss
                    d_loss = d_loss_blur + d_loss_gray

                    g_loss_total += g_loss
                    d_loss_total += d_loss

                    # Generator
                    gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
                    self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

                    # Discriminator
                    gradients_of_discriminator = disc_tape.gradient(d_loss, self.disc_sn.trainable_variables)
                    self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.disc_sn.trainable_variables))

                    pbar.set_description("Epoch {} || g_loss: {} || d_loss: {}".format(epoch,
                                                                                       g_loss,
                                                                                       d_loss))
                    if iterator % 200 == 0:
                        results = self.generator(input_photo)
                        results = guided_filter(input_photo, results, r=4)
                        saved_test = os.path.join(HOME, "test_images")
                        os.makedirs(saved_test, exist_ok=True)
                        write_batch_image(results, saved_test, f"{iterator}_result.jpg", 1)

            if epoch % 5 == 0:
                self.ckpt_gen_manager.save()
                self.ckpt_disc_manager.save()

            print("Epoch {} || g_loss_total: {} || d_loss_total: {}".format(epoch,
                                                                            g_loss_total / min_len,
                                                                            d_loss_total / min_len))


if __name__ == '__main__':
    pc = False
    if pc:
        real_face = "dataset/face_photo"
        cartoon_faces = "dataset/face_cartoon"
        real_scenery = "dataset/scenery_photo"
        cartoon_scenery = "dataset/scenery_cartoon"
        train = Trainer(real_face, cartoon_faces,
                        real_scenery, cartoon_scenery,
                        image_shape=128, epochs=5,
                        batch_size=4, channels=16, use_parallel=True, retrain=False)
    else:
        real_face = "/content/drive/MyDrive/dataset/face_photo"
        cartoon_faces = "/content/drive/MyDrive/dataset/face_cartoon"
        real_scenery = "/content/drive/MyDrive/dataset/scenery_photo"
        cartoon_scenery = "/content/drive/MyDrive/dataset/scenery_cartoon"
        train = Trainer(real_face, cartoon_faces,
                        real_scenery, cartoon_scenery,
                        image_shape=256, epochs=50,
                        batch_size=16, channels=16, use_parallel=True)

    train.train_step()
