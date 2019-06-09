import os
import random
import shutil
import tarfile
import cv2
import numpy as np
from keras.utils import Sequence

from utilities import download_file, download_image_cv2_urllib


class DataGen(Sequence):
    """
    This generator downloads one tar file at each epoch. Extracts and selects the valid images from it to
    form batches. And after the epoch is complete, deletes the files to free up space.
    """
    def __init__(self, valid_ids_dict, num_classes, start=10, batch_size=128, steps=10, verbose=1):

        self.valid_ids_dict = valid_ids_dict # dict of image ids to landmarks {image_id: landmark_id}
        self.NUM_CLASSES = num_classes # number of valid classes to consider

        self.batch_size = batch_size
        self.steps = steps # should be equal to the number of epochs
        self.images = []
        self.landmarks = []
        self.tar_idx = start
        self.epoch_init()

    def epoch_init(self):
        self.all_images = []
        self.all_landmarks = []

        if self.tar_idx < 10:
            tarfilestr = "00" + str(self.tar_idx)
        elif self.tar_idx < 100:
            tarfilestr = "0" + str(self.tar_idx)
        else:
            tarfilestr = str(self.tar_idx)

        download_file("https://s3.amazonaws.com/google-landmark/train/images_{}.tar".format(tarfilestr), "images.tar",
                      bar=False)
        #print(os.listdir())
        tar = tarfile.open('images.tar')
        tar.extractall("imagesfolder")
        tar.close()

        self.total = self.pickfiles("imagesfolder")
        self.tar_idx += 1
        print("tar", self.tar_idx - 1, "total:", self.total)

    def pickfiles(self, dirr):
        count = 0
        for f in os.listdir(dirr):
            if os.path.isfile(dirr + "/" + f):
                if f[:-4] in self.valid_ids_dict:
                    self.all_images.append(dirr + "/" + f)
                    self.all_landmarks.append(self.valid_ids_dict[f[:-4]])
                    count += 1
            else:
                count += self.pickfiles(dirr + "/" + f)
        return count

    def normalize(self, data):
        return data / 255 - 0.5

    def __getitem__(self, index):
        image_path_list = self.all_images[index * self.batch_size:min(self.total, (index + 1)) * self.batch_size]
        class_list = self.all_landmarks[index * self.batch_size:min(self.total, (index + 1)) * self.batch_size]

        if len(image_path_list) == 0:
            image_path_list = self.all_images[:self.batch_size]
            class_list = self.all_landmarks[:self.batch_size]

        images = []
        y_list = []
        for ix in range(len(image_path_list)):
            try:
                image_path = image_path_list[ix]
                im = cv2.imread(image_path)
                im = cv2.resize(im, (192, 192), interpolation=cv2.INTER_AREA)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                if im.shape == (192, 192, 3):
                    images.append(im)
                    y_list.append(class_list[ix])
            except:
                continue

        x = np.array(images)
        y = np.zeros((len(y_list), self.NUM_CLASSES))

        for i in range(len(y_list)):
            y[i, y_list[i]] = 1.

        return x, y

    def on_epoch_end(self):
        self.steps -= 1
        os.unlink("images.tar")
        shutil.rmtree("imagesfolder")
        if self.steps > 0:
            self.epoch_init()

    def __len__(self):
        return self.total // self.batch_size + int(self.total % self.batch_size > 0)


class DataGenURLVersion(Sequence):
    """
    This generator uses the image urls from the train dataset to form batches
    and downloads each image individually. It will be approx 10 times slower than above version.
    """
    def __init__(self, valid_urls_dict, num_classes, data, batch_size=24, verbose=1):
        self.batch_size = batch_size
        self.data_urls = data
        self.NUM_CLASSES = num_classes # number of classes
        self.valid_urls_dict = valid_urls_dict # dict of url and corresponding landmark {image_url: landmark}

    def normalize(self, data):
        return data

    def __getitem__(self, index):
        batch_urls = random.sample(self.data_urls, self.batch_size)

        output = []
        y_classes = []
        for url in batch_urls:
            im = download_image_cv2_urllib(url)
            if im.size != 0:
                output.append(im)
                y_classes.append(self.valid_urls_dict[url.split("/")[-1]])

        x = np.array(output)
        y = np.zeros((len(output), self.NUM_CLASSES))

        for i in range(len(y_classes)):
            y[i, y_classes[i]] = 1.

        return x, y

    def on_epoch_end(self):
        return

    def __len__(self):
        # return len(valid_urls_list) // self.batch_size
        return 10