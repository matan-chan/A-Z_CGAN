from os.path import isfile, join, isdir
from os import listdir
import numpy as np
import threading
import zipfile
import cv2
import os


class DataPipe:
    """
    This class handling the data.

    Attributes
    ----------

    all_images : list
        list of all the images

    """

    def __init__(self):
        self.all_images = []
        self.all_labels = []

    @staticmethod
    def agent_loader(start, end, all_images, result, index, dir):
        """
        thread function which load images
        :param start:the start index of the thread`s section
        :type start: int
        :param end:the end index of the thread`s section
        :type end: int
        :param all_images:list of images full paths
        :type all_images: list
        :param result:list of images
        :type result: list
        :param index:the thread`s` number
        :type index: int
        """
        print(dir, start, ' is on')
        arr = []
        for i in range(start, end):
            img = cv2.cvtColor(cv2.imread(dir + "/" + all_images[i]), cv2.COLOR_BGR2GRAY)
            if img.size == 784:
                img = cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            arr.append(img)
        arr = np.array(arr)
        arr = (arr.astype(np.float32) - 127.5) / 127.5
        result[index] = list(arr)

    def load_all_images(self):
        """
        loads all the training images
        """
        threads = [None] * os.cpu_count()
        results = [None] * os.cpu_count()
        # if the is`nt "data" directory and there is "data.zip" this section will extract it
        if not isdir('data'):
            if isfile('data.zip'):
                with zipfile.ZipFile('data.zip', 'r') as zip_ref:
                    zip_ref.extractall('')
            else:
                print('no "data" directory')
                return np.array([])

        for index, dir in enumerate(listdir('data')):
            all_images = [f for f in listdir('data/' + dir) if isfile(join('data/' + dir, f))]
            part = len(all_images) // os.cpu_count()
            for i in range(len(threads)):
                too = len(all_images) if i + 1 == len(threads) else part * (i + 1)
                threads[i] = threading.Thread(target=self.agent_loader,
                                              args=(part * i, too, all_images, results, i, 'data/' + dir))
                threads[i].start()
            for i in range(len(threads)):
                threads[i].join()
            final = []
            for i in range(len(threads)):
                final = final + results[i]
            self.all_images += final
            self.all_labels += np.full((len(final),), index).tolist()

    def get_butch(self, butch_size):
        """
        gets a butch of training images can be called right out *slow but ram efficient* or use "load_all_images"
        before and it will be *quick but ram hungry*
        """
        if self.all_images:
            images = []
            labels = []
            idx = np.random.randint(0, len(self.all_images), butch_size)
            for i in idx:
                images.append(self.all_images[i])
                labels.append(self.all_labels[i])
            return np.array(images), np.array(labels)
        # if the is`nt "data" directory and there is "data.zip" this section will extract it
        if not isdir('data'):
            if isfile('data.zip'):
                with zipfile.ZipFile('data.zip', 'r') as zip_ref:
                    zip_ref.extractall('')
            else:
                print('no "data" directory')
                return np.array([])
