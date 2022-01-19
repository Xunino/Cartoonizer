import tensorflow as tf

from utils.utils import get_list_images


class DataLoader:
    def __init__(self, image_paths, image_shape=256, batch_size=32):
        self.image_paths = image_paths
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.autotune = tf.data.AUTOTUNE
        self.list_image_path = get_list_images(self.image_paths)
        self.next_batch = 0

        assert len(self.list_image_path) != 0

    def __len__(self):
        return len(self.list_image_path)

    def processing_image(self, file_image):
        img = tf.io.read_file(file_image)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [self.image_shape, self.image_shape])
        img = img / 127.5 - 1
        return img

    def config_for_image_performance(self, ds):
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.autotune)
        ds = ds.shuffle(buffer_size=42)
        return ds

    def __call__(self, *args, **kwargs):
        self.train_ds = tf.data.Dataset.list_files(self.list_image_path)
        self.train_ds = self.train_ds.map(self.processing_image, num_parallel_calls=self.autotune)
        # Performance
        return self.config_for_image_performance(self.train_ds)


if __name__ == '__main__':
    samples = "dataset/face_cartoon"
    loader = DataLoader(samples)
    print(next(iter(loader())))
