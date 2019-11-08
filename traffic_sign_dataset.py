import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle
from graph import Graph
import albumentations as a

'''
The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
'''


class TrafficData():

    def __init__(self):
        self.training_file = "data/train.p"
        self.validation_file = "data/valid.p"
        self.testing_file = "data/test.p"
        self.classes_file = "signnames.csv"

        self.augment = a.Compose([
            a.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=20, p=1),
            a.RandomBrightness(0.2, p=1)
        ])
        self.input_shape = None

        self.train = None
        self.test = None
        self.valid = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_valid = None
        self.y_valid = None
        self.class_names = None

        self.load_data()

    def load_data(self):
        '''
        Method loads dataset and classes names.
        :return: None
        '''
        with open(self.training_file, mode='rb') as f:
            self.train = pickle.load(f)
        with open(self.validation_file, mode='rb') as f:
            self.valid = pickle.load(f)
        with open(self.testing_file, mode='rb') as f:
            self.test = pickle.load(f)

        self.X_train, self.y_train = self.train['features'], self.train['labels']
        self.X_valid, self.y_valid = self.valid['features'], self.valid['labels']
        self.X_test, self.y_test = self.test['features'], self.test['labels']
        self.input_shape = self.train['sizes'][0]

        self.class_names = pd.read_csv(self.classes_file).set_index('ClassId').to_dict()['SignName']

    def normalize_data(self, brightness: bool = False, grayscale: bool = False):
        '''
        Normalize entire dataset from 0..255 to 0.0 .. 1.0
        Additionally can improve brightness as well as change image to grayscale.
        :param brightness: If brightness shall be corrected.
        :param grayscale: If color map shall be changed to grascale.
        :return:
        '''
        self.X_train = self.__normalize__(self.X_train, brightness, grayscale)
        self.X_test = self.__normalize__(self.X_test, brightness, grayscale)
        self.X_valid = self.__normalize__(self.X_valid, brightness, grayscale)

    def __normalize__(self, data, brightness: bool = False, grayscale: bool = False):
        '''
        Normalize entire dataset from 0..255 to 0.0 .. 1.0
        Additionally can improve brightness as well as change image to grayscale.
        :param data: Dataset to be normalized.
        :param brightness: If brightness shall be corrected.
        :param grayscale: If color map shall be changed to grascale.
        :return:
        '''
        X_array = []
        channels = 3
        for image in data:
            if brightness:
                image = Graph.adjust_brightness(image)
                pass
            if grayscale:
                image = Graph.to_grayscale(image)
                channels = 1
                pass
            new_image = np.divide(image, 255.)
            X_array.append(new_image)

        result = np.asarray(X_array)
        return result.reshape(result.shape[0], 32, 32, channels)

    def shuffle_dataset(self):
        '''
        Method shuffles dataset.
        '''
        self.X_train, self.y_train = shuffle(self.X_train, self.y_train)
        self.X_valid, self.y_valid = shuffle(self.X_valid, self.y_valid)
        self.X_test, self.y_test = shuffle(self.X_test, self.y_test)

    def get_training_dataset(self):
        return self.X_train, self.y_train

    def get_validation_dataset(self):
        return self.X_valid, self.y_valid

    def get_testing_dataset(self):
        return self.X_test, self.y_test

    def __print_class_names(self):
        print(self.class_names)

    def preview_random(self, grayscale: bool = False):
        '''
        Method plots 12 random images from training dataset with its label.
        :return: None
        '''
        f, axes = plt.subplots(4, 3, sharey=True, figsize=(8, 8))
        for x in range(0, 4):
            for y in range(0, 3):
                image_index = random.randint(0, len(self.X_train) - 1)
                image = self.X_train[image_index]
                label = self.class_names[self.y_train[image_index]]
                axes[x, y].set_title(label)
                axes[x, y].axis('off')
                if grayscale:
                    img = np.reshape(image, (32, 32))
                    axes[x, y].imshow(img, cmap='gray')
                else:
                    axes[x, y].imshow(image)
                # print(x, ', ', y, ', ', image.shape, ', ', image_index, ', ', label)
        plt.tight_layout()
        plt.show()

    def label_for(self, class_id: int = None):
        if class_id is not None:
            return self.class_names[class_id]
        pass

    def augment_dataset(self):
        aug_x_train = []
        [aug_x_train.append(self.__augment__(img)) for img in self.X_train]

        self.X_train = np.append(self.X_train, aug_x_train, axis=0)
        self.y_train = np.append(self.y_train, self.y_train, axis=0)

    def __augment__(self, img):
        return self.augment(image=img)['image']
