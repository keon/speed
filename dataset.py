from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from preprocess import optical_flow, aug


class Dataset:
    def __init__(self,
                 folder:str,
                 resize:(int, int),
                 batch_size:int,
                 timesteps:int,
                 windowsteps:int,
                 shift:int,
                 train:bool):
        self.folder  = folder
        self.resize = resize
        self.batch_size = batch_size
        self.timesteps  = timesteps
        self.train = train
        self.images = sorted(os.listdir(folder + 'images/'))
        self.labels = open(folder + 'labels.txt').readlines()
        self.data = self._sliding_window(self.images, shift, windowsteps)

    def _sliding_window(self, images, shift, windowsteps):
        """ slide window by the windowsteps to make dataset """
        window = []
        num_windows = len(images) - self.timesteps + 1
        for i in range(shift, num_windows, windowsteps):
            if self.train:
                labels = [float(label)
                    for label in self.labels[i:i+self.timesteps]]
                window.append([images[i:i+self.timesteps], labels[-1]])
            else:
                window.append([images[i:i+self.timesteps], None])
        return window

    def get_batcher(self, shuffle=True, augment=True):
        """ produces batch generator """
        w, h = self.resize

        if shuffle: np.random.shuffle(self.data)
        data = iter(self.data)
        while True:
            x = np.zeros((self.batch_size, self.timesteps, h, w, 3))
            y = np.zeros((self.batch_size, 1))
            for b in range(self.batch_size):
                images, label = next(data)
                for t, img_name in enumerate(images):
                    image_path = self.folder + 'images/' + img_name
                    img = cv2.imread(image_path)
                    img = img[190:350, 100:520] # crop
                    if augment:
                        img = aug.augment_image(img) # augmentation
                    img = cv2.resize(img.copy(), (w, h))
                    x[b, t] = img
                y[b] = label
            x = np.transpose(x, [0, 4, 1, 2, 3])
            yield x, y
