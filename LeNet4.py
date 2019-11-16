import tensorflow as tf
import neptune
from tensorflow.contrib.layers import flatten
from traffic_sign_dataset import TrafficData
import tensorflow.contrib.slim as slim
from sklearn.utils import shuffle
import numpy as np


class LeNet4():
    def __repr__(self):
        return 'LeNet4()'

    def __init__(self, output_classes: int = 43, channels: int = 3):
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        self.my_name = 'LeNet v4'
        self.model_file_name = './lenet'
        self.mu = 0
        self.sigma = 0.1
        self.epochs = 10
        self.batch_size = 128
        self.learn_rate = 0.001
        self.x = tf.placeholder(tf.float32, (None, 32, 32, channels))
        self.y = tf.placeholder(tf.int32, (None))
        self.one_hot_y = tf.one_hot(self.y, output_classes)

        self.saver = None
        self.dataset = None
        self.accuracy_operation = None
        self.cross_entropy = None
        self.loss_operation = None
        self.optimizer = None
        self.training_operation = None
        self.correct_prediction = None
        self.log_neptune = False

        # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
        self.lay1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=self.mu, stddev=self.sigma))
        self.lay1_b = tf.Variable(tf.zeros([6]))
        padding = 'VALID'
        self.layer1 = tf.nn.conv2d(self.x, self.lay1_W, strides=[1, 1, 1, 1], padding=padding) + self.lay1_b

        # TODO: Activation.
        self.layer1 = tf.nn.relu(self.layer1)

        # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
        ksize = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        self.layer1 = tf.nn.max_pool(self.layer1, ksize, strides, padding)

        # TODO: Layer 2: Convolutional. Output = 10x10x16.
        self.lay2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=self.mu, stddev=self.sigma))
        self.lay2_b = tf.Variable(tf.zeros([16]))
        padding = 'VALID'
        strides = [1, 1, 1, 1]
        self.layer2 = tf.nn.conv2d(self.layer1, self.lay2_W, strides, padding) + self.lay2_b

        # TODO: Activation.
        self.layer2 = tf.nn.relu(self.layer2)

        # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
        ksize = [1, 2, 2, 1]
        strides = [1, 2, 2, 1]
        padding = 'VALID'
        self.layer2 = tf.nn.max_pool(self.layer2, ksize, strides, padding)

        # TODO: Flatten. Input = 5x5x16. Output = 400.
        self.flat = flatten(self.layer2)

        # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
        self.lay3_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=self.mu, stddev=self.sigma))
        self.lay3_b = tf.Variable(tf.zeros([120]))
        self.layer3 = tf.matmul(self.flat, self.lay3_W) + self.lay3_b

        # TODO: Activation.
        self.layer3 = tf.nn.relu(self.layer3)

        # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
        self.lay4_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=self.mu, stddev=self.sigma))
        self.lay4_b = tf.Variable(tf.zeros([84]))
        self.layer4 = tf.matmul(self.layer3, self.lay4_W) + self.lay4_b

        # TODO: Activation.
        self.layer4 = tf.nn.relu(self.layer4)

        # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
        self.lay5_W = tf.Variable(tf.truncated_normal(shape=(84, output_classes), mean=self.mu, stddev=self.sigma))
        self.lay5_b = tf.Variable(tf.zeros([output_classes]))
        self.layer5 = tf.matmul(self.layer4, self.lay5_W) + self.lay5_b

        self.network = self.layer5

    def start_neptune_session(self, api_token, prj_name):
        assert api_token is not None
        assert prj_name is not None
        neptune.init(
            api_token=api_token,
            project_qualified_name=prj_name)
        self.log_neptune = True

    def get_network(self):
        '''
        :return: Tensor with entire NN architecture.
        '''
        return self.network

    def set_hiperparams(self, epochs: int = 10, batch_size: int = 64, learn_rate: float = 0.002):
        '''
        Method sets hiperparameters.

        :param epochs: Number of epochs for training.
        :param batch_size: Size of batch.
        :param learn_rate: Learning rate.
        :return:
        '''
        self.epochs = epochs
        self.batch_size = batch_size
        self.learn_rate = learn_rate

    def evaluate(self, X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, self.batch_size):
            batch_x, batch_y = X_data[offset:offset + self.batch_size], y_data[offset:offset + self.batch_size]
            accuracy = sess.run(self.accuracy_operation, feed_dict={self.x: batch_x, self.y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def train(self, dataset: TrafficData = None, neptune_tags = None):
        assert dataset is not None
        self.dataset = dataset
        experiment = None

        if self.log_neptune:
            experiment = neptune.create_experiment(name=self.my_name, params={'batch_size': self.batch_size,
                                                                              'lr': self.learn_rate,
                                                                              'nr_epochs': self.epochs})
            experiment.append_tag(self.my_name)
            if neptune_tags is not None:
                [experiment.append_tag(tag) for tag in neptune_tags]

        logits = self.get_network()
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y, logits=logits)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
        self.training_operation = self.optimizer.minimize(self.loss_operation)

        self.correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x_train, y_train = self.dataset.get_training_dataset()
            x_valid, y_valid = self.dataset.get_validation_dataset()
            x_test, y_test = self.dataset.get_testing_dataset()
            num_examples = len(x_train)

            print("Training...")
            print()
            for i in range(self.epochs):
                x_train, y_train = shuffle(x_train, y_train)
                for offset in range(0, num_examples, self.batch_size):
                    end = offset + self.batch_size
                    batch_x, batch_y = x_train[offset:end], y_train[offset:end]
                    sess.run(self.training_operation, feed_dict={self.x: batch_x, self.y: batch_y})

                validation_accuracy = self.evaluate(x_valid, y_valid)
                test_accuracy = self.evaluate(x_test, y_test)

                print("EPOCH {} ...".format(i + 1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print("Test Accuracy = {:.3f}".format(test_accuracy))
                print()
                if self.log_neptune:
                    experiment.send_metric('validation_accuracy', validation_accuracy)
                    experiment.send_metric('test_accuracy', test_accuracy)

            self.saver.save(sess, self.model_file_name)
            if self.log_neptune:
                experiment.stop()

    def predict(self, images):
        '''
        Method runs inception method for trained model with given set of images.
        Method returns label for highest prediction class per each image.
        :param images: Set of images.
        :return: label for highest prediction class per each image.
        '''
        assert images is not None

        with tf.Session() as sess:
            # Restore variables from disk.
            self.saver.restore(sess, self.model_file_name)

            results = self.network.eval(feed_dict={self.x: images})
            values = np.argmax(results, axis=1)
            labels = [self.dataset.label_for(val) for val in values]
            return labels

    def predict_image(self, image):
        assert image is not None

        image = np.reshape(image, [32, 32, 1])
        with tf.Session() as sess:
            # Restore variables from disk.
            self.saver.restore(sess, self.model_file_name)

            results = self.network.eval(feed_dict={self.x: [image]})
            # values = np.argmax(results, axis=1)
            # labels = [self.dataset.label_for(val) for val in values]
            return results

    def model_summary(self):
        '''
        Method prints summary of the NN model.
        :return:
        '''
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)