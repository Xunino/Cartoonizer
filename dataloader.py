import cv2
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


class DataLoader:
    def __init__(self, filename_list, image_shape=256, batch_size=32):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.next_batch = 0
        self.filename_list = sorted(filename_list)

    def __len__(self):
        return len(self.filename_list)

    def __call__(self):
        assert len(self.filename_list) != 0
        batch_data = []
        self.filename_list = shuffle(self.filename_list, random_state=42)
        batch = self.filename_list[self.next_batch: self.next_batch + self.batch_size]
        for i in range(len(batch)):
            image = cv2.imread(batch[i])
            image = cv2.resize(image, dsize=(self.image_shape, self.image_shape), interpolation=cv2.INTER_AREA)
            h, w, c = np.shape(image)
            h, w = (h // 8) * 8, (w // 8) * 8
            image = image[:h, :w, :]
            image = image.astype(np.float32) / 127.5 - 1
            batch_data.append(image)

        self.next_batch += self.batch_size
        if self.next_batch > self.__len__():
            self.next_batch = 0

        return np.asarray(batch_data)


class DataLoaderWithTF:
    def __init__(self, get_list_images, image_shape=256, batch_size=32):
        self.list_image_path = sorted(get_list_images)
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.autotune = tf.data.AUTOTUNE
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
    from utils.utils import get_list_images

    samples = "dataset/face_cartoon"
    loader = DataLoaderWithTF(get_list_images(samples))
    for i in range(10):
        print(loader())
