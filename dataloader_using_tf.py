import tensorflow as tf


class DataLoader:
    def __init__(self, image_paths, image_shape=256, batch_size=32):
        self.image_paths = image_paths
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.autotune = tf.data.AUTOTUNE
        self.list_image_path = tf.data.Dataset.list_files(self.image_paths)
        assert len(self.list_image_path) != 0

    def __len__(self):
        return len(self.list_image_path)

    def config_for_text_performance(self, ds):
        ds = tf.data.Dataset.from_tensor_slices(ds)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=self.autotune)
        return ds

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
        self.train_ds = self.list_image_path.map(self.processing_image, num_parallel_calls=self.autotune)
        # Performance
        return self.config_for_image_performance(self.train_ds)
