import pickle
import numpy as np
from sklearn.utils import shuffle

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

        self.load_data()

    def load_data(self):
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

    def normalize_data(self):
        '''
        Normalize entire data from 0..255 to 0.0 .. 1.0
        '''
        self.X_train = self.__normalize__(self.X_train)
        self.X_test = self.__normalize__(self.X_test)
        self.X_valid = self.__normalize__(self.X_valid)

    def __normalize__(self, data):
        '''
        Normalize data from 0..255 to 0.0 .. 1.0
        :param data: dataset to be normalized.
        :return Normalized dataset.
        '''
        X_array = []
        for image in data:
            X_array.append(np.divide(image, 255.))
        return np.asarray(X_array)

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
