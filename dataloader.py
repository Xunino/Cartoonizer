import os
import cv2
import numpy as np


class DataLoader:
    def __init__(self, path_images, image_shape=256, batch_size=32):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.next_batch = 0
        self.filename_list = []
        self.get_path(path_images)

    def get_path(self, path_images):
        for name in os.listdir(path_images):
            self.filename_list.append(os.path.join(path_images, name))
        self.filename_list.sort()

    def __len__(self):
        return len(self.filename_list)

    def run(self):
        assert len(self.filename_list) != 0
        batch_data = []
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

        return np.asarray(batch_data), self.__len__()


if __name__ == '__main__':
    samples = "dataset/faces"
    loader = DataLoader(samples).run()
    print(loader.shape)
