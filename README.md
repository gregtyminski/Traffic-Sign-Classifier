# Project: Build a Traffic Sign Recognition Program

[//]: # (Image References)
[step-0-preview-data]: ./images/step-0-preview-data.png "Initial dataset preview"
[step-3-preview-data]: ./images/step-3-preview-data.png "Dataset with grayscale and brightness correction"
[step-1-lenet-best1]: ./images/step-1-lenet-best-part1.png "LeNet - best results"
[step-1-lenet-best2]: ./images/step-1-lenet-best-part2.png "LeNet - best results"
[step-1-lenet-t2-1]: ./images/step-1-lenet-t2-1.png "LeNet - graph with best results"
[step-1-lenet-t2-2]: ./images/step-1-lenet-t2-2.png "LeNet - graph with best results"
[dataset-histogram]: ./images/dataset-histogram.png "Dataset histogram"
[step-2-lenet-t1]: ./images/step-2-lenet-t1.png "LeNet 2 - dropout results"
[step-2-lenet-t1-graph]: ./images/step-2-lenet-t1-graph.png "LeNet 2 - graph"
[step-3-lenet-t1]: ./images/step-3-lenet-t1.png "LeNet 3 - dropout results with improved brightness and grayscale"
[step-3-lenet-t1-graph]: ./images/step-3-lenet-t1-graph.png "LeNet 3 - graph with improved brightness and grayscale"
[step-4-lenet-t1]: ./images/step-4-lenet-t1.png "LeNet 4 - results with improved brightness and grayscale"
[step-4-lenet-t1-graph]: ./images/step-4-lenet-t1-graph.png "LeNet 4 - with improved brightness and grayscale"
[step-5-lenet-t1]: ./images/step-5-lenet-t1.png "LeNet 4 - results with data augmentation"
[step-5-lenet-t1-graph]: ./images/step-5-lenet-t1-graph.png "LeNet 4 - with data augmentation"
[step-5-lenet-test-data]: ./images/step-5-lenet-test-data.png "LeNet 4 - results with data augmentation on test dataset"
[test-img-1]: ./input/sign_1.png "Test Sign 1"
[test-img-2]: ./input/sign_2.png "Test Sign 2"
[test-img-3]: ./input/sign_3.png "Test Sign 3"
[test-img-4]: ./input/sign_4.png "Test Sign 4"
[test-img-5]: ./input/sign_5.png "Test Sign 5"
[test-img-6]: ./input/sign_6.png "Test Sign 6"
[test-img-7]: ./input/sign_7.png "Test Sign 7"
[test-img-8]: ./input/sign_8.png "Test Sign 8"
[test-img-9]: ./input/sign_9.png "Test Sign 9"
[test-img-10]: ./input/sign_10.png "Test Sign 10"

This project is a part of:  
 [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

In this project we are going to train a network to recognize traffic signs.<br>
The dataset of traffic signs come from [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and this dataset will be used to train the neural network recognizing traffic signs.<br>
This dataset is to large to be kept in GitHub, therefore if you want to run any training job please download the dataset from [this link](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip) and unpack it to `data` directory.<br>
<br>
To track results of all training experiments I use [Neptune.ml](http://neptune.ml) tool.

## Step 0: Load The Data

At the beginning we need to load the dataset. Dataset is kept in the folder `data` and it contains following files:\

```bash
total 311760
-rw-r--r--@ 1 grzegorz.tyminski  staff   38888118 Nov  7  2016 test.p
-rw-r--r--@ 1 grzegorz.tyminski  staff  107146452 Feb  2  2017 train.p
-rw-r--r--@ 1 grzegorz.tyminski  staff   13578712 Feb  2  2017 valid.p
```

Class loading this dataset is implemented in the file [traffic_sign_dataset.py](traffic_sign_dataset.py).<br>
To load data you need to run following code:

```python
from traffic_sign_dataset import TrafficData

dataset = TrafficData() # dataset is loaded from files while creating instance of the `TrafficData` object.

dataset.normalize_data() # dataset is normalized --> values are in range 0..1 instead of 0..255
dataset.shuffle_dataset() # dataset is shuffled

X_train, y_train = dataset.get_training_dataset() # get training part of dataset
X_valid, y_valid = dataset.get_validation_dataset() # get validation part of dataset
X_test, y_test = dataset.get_testing_dataset() # get testing part of dataset
```

#### Dataset preview

When the dataset is loaded, we can preview some random images from `train` dataset just by calling `dataset.preview_random()` method.<br>
We will obtain something like:<br>
![alt text][step-0-preview-data]

#### Dataset histogram

Let's have a look on the histogram of classes in dataset.
![alt text][dataset-histogram]
We can see here, that some of the classes in dataset have 10x less pictures than the other ones. These are:

```text
Speed limit (20km/h)
Speed limit (100km/h)
Vehicles over 3.5 metric tons prohibited
Dangerous curve to the left
Road narrows on the right
Pedestrians
Bicycles crossing
End of all speed and passing limits
Go straight or left
Keep left
End of no passing
```

#### Dataset normalization

In this example we can clearly see, that some of images are very dark. Probably too dark for humans eye to recognize the sign.
<br>
The class `TrafficData` contains method `normalize_data()` to normalize the dataset (i.e. change values of pixels from 0..255 to 0..1) as well as the method `shuffle_dataset()` to randomize the order of images in dataset.

## Step 1: Train LeNet network

In the very first step we are going to train the same LeNet neural network on as is the MNIST example.<br>
The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point.
Let's just verify, how good it is.<br>
Class `LeNet` with model architecture is defined in the [LeNet.py](LeNet.py) file.<br>
It's input and output shape is adjusted for Traffic Sign Dataset (3-channel color with 43-classes output).<br>
I've run grid search over hyperparams with values for `epoch` _(10 or 15)_, `batch_size` _(32, 64, 128, 256, 512, 1024)_ and `learn_rate` _(0.0005, 0.001, 0.002)_
For several trials we got the validation accuracy up to __0.953968__ for `epochs` equal to 10, `batch_size` equal to 64 and `learn_rate` equal to 0.002<br>
![alt text][step-1-lenet-best1]<br>
<br>
![alt text][step-1-lenet-best2]
The results did not differ much for several hyperparams. However we can clearly show, that smaller `batch_size` resulted in better result. Values for `learn_rate` below _0.001_ did not give satisfying results and the number of `epochs` bigger than 10 gave no better results (see graph below).
![alt text][step-1-lenet-t2-1]<br>
<br>
![alt text][step-1-lenet-t2-2]<br>
This `LeNet` model has following architecture:<br>

| Variables:                            | name        | type shape | size                   |
| ------------------------------------- | ----------- | ---------- | ---------------------- |
| Variable:0                            | float32_ref | 5x5x3x6    | [450, bytes: 1800]     |
| Variable_1:0                          | float32_ref | 6          | [6, bytes: 24]         |
| Variable_2:0                          | float32_ref | 5x5x6x16   | [2400, bytes: 9600]    |
| Variable_3:0                          | float32_ref | 16         | [16, bytes: 64]        |
| Variable_4:0                          | float32_ref | 400x120    | [48000, bytes: 192000] |
| Variable_5:0                          | float32_ref | 120        | [120, bytes: 480]      |
| Variable_6:0                          | float32_ref | 120x84     | [10080, bytes: 40320]  |
| Variable_7:0                          | float32_ref | 84         | [84, bytes: 336]       |
| Variable_8:0                          | float32_ref | 84x43      | [3612, bytes: 14448]   |
| Variable_9:0                          | float32_ref | 43         | [43, bytes: 172]       |
Total size of variables: __64811__<br>
Total bytes of variables: __259244__<br>

## Step 2: Modify LeNet network

We could modify a bit `LeNet` network and try it on this dataset. First modification would be to add `dropout` in the network. Modified `LeNet` is implemented in the file `LeNet2.py`.<br>
I've added it in the network first 1 and later 2 dropouts and run grid search over hyperparams in both variants including also dropout value.
First dropout has been added between 2nd and 3rd layer in network:

```python
# layer 2
self.flat = flatten(self.layer2)
# dropout
self.flat = tf.nn.dropout(self.flat, self.dropout_val)
# layer 3
self.lay3_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=self.mu, stddev=self.sigma))
self.lay3_b = tf.Variable(tf.zeros([120]))
self.layer3 = tf.matmul(self.flat, self.lay3_W) + self.lay3_b
```

... and the second has been added between 4th and 5th layer in network:

```python
# layer 4
self.layer4 = tf.nn.relu(self.layer4)
# dropout
self.layer4 = tf.nn.dropout(self.layer4, self.dropout_val)
# layer 5
self.lay5_W = tf.Variable(tf.truncated_normal(shape=(84, output_classes), mean=self.mu, stddev=self.sigma))
self.lay5_b = tf.Variable(tf.zeros([output_classes]))
self.layer5 = tf.matmul(self.layer4, self.lay5_W) + self.lay5_b
```

Here are results from training jobs:
![alt text][step-2-lenet-t1]<br>
<br>
... and when we plot the results of trainings:
![alt text][step-2-lenet-t1-graph]<br>
<br>
We have actually got worse results (no matter if single droput or double ones and the value of dropout).

## Step 3: Better normalization

Let's modify the dataset. There are 2 potential improbements:

- improve brightness
- change color scale from 3-channel to 1-channel (grayscale)

Brightness correction as well as the change to grayscale color map is implemented in the `TrafficData` class. To include these 2 steps in dataset just 2 params: `brighness` and `grayscale` in dataset normalization need to be added:

```python
# initiate and load dataset
dataset = TrafficData()

# normalize dataset --> change values of pixels from 0..255 to 0..1 & include brightness correction as well as change to grayscale color map
dataset.normalize_data(brightness=True, grayscale=True)
```

Normalize dataset looks as follows at the moment:<br>
![alt text][step-3-preview-data]<br>

#### Network with dropout

As dataset input shape has changed, we had to change the network architecture to adapt this input shape. New version of the network __including__ dropout has been implemented in `LeNet3.py`.<br>
Training network __with__ dropout gave no result improvement (__0.904535__):'
![alt text][step-3-lenet-t1]<br>
<br>
... and when we plot the results of trainings:
![alt text][step-3-lenet-t1-graph]<br>

#### Network without dropout

Training network __without__ dropout (`LeNet4.py`) gave similar best result (__0.944898__) on validation dataset as in dataset without brithness improvements and grayscaled, but the worst result was significantly better (__0.903855__) than previously:'
![alt text][step-4-lenet-t1]<br>
<br>
... and when we plot the results of trainings:
![alt text][step-4-lenet-t1-graph]<br>

## Step 4: Improve dataset

If we have a look on the dataset classes distribution (part `Step 0: Load The Data`), we clearly see, that distribution of classes is not equal with big differences. The dataset requires improvement --> __augmentation__.<br>
Very simple augmentation is implemented in the `TrafficData` class. To double the train dataset we need to just call the method `augment_dataset()` after loading it and before normalization and before shuffling.'

```python
from traffic_sign_dataset import TrafficData

# initiate and load dataset
dataset = TrafficData()
# augment dataset to get 2x bigger dataset
dataset.augment_dataset()
# augment dataset again to get 4x bigger dataset
dataset.augment_dataset()

# normalize dataset --> change values of pixels from 0..255 to 0..1
dataset.normalize_data(brightness=True, grayscale=True)
# randomize the order of images in dataset
dataset.shuffle_dataset()
```

The library [Albumentations](https://github.com/albu/albumentations) is used for data augmentation. Single augmentation step `dataset.agument_dataset()` just adds 1 new image for each already existing in training dataset which is slightly rotated, slightly shifted and slightly brightness modified. As a result 2x bigger dataset is received. As we call it twice, we get 4x bigger dataset with modified images.<br>
<br>
When we run training job (`LeNet4` network with grayscaled images and without dropout) we have received much better accuracy result __0.971429__<br>
![alt text][step-5-lenet-t1]<br>
<br>
... and when we plot the results of trainings:
![alt text][step-5-lenet-t1-graph]<br>
<br>
The result on `test` dataset for this training was: __0.950119__<br>
![alt text][step-5-lenet-test-data]<br>

## Test performance of model on random images from internet

Last step is to verify the model on random images of german signs downloaded from internet. These pictures are:<br>

| ID  | Image                   | Top 3 probabilities | Corresponding labels |
| --- | ----------------------- | ------------------------ | -------------------- |
| 1.  | ![alt text][test-img-1] | prob=0.80<br>prob=0.12<br>prob=0.03 | Speed limit (30km/h)<br>Roundabout mandatory<br>Speed limit (70km/h)| 
| 2.  | ![alt text][test-img-2] | prob=1.00<br>prob=0.00<br>prob=0.00 | Turn right ahead<br>Ahead only<br>Right-of-way at the next intersection |
| 3.  | ![alt text][test-img-3] | prob=0.99<br>prob=0.01<br>prob=0.00 | Beware of ice/snow<br>Right-of-way at the next intersection<br>Bicycles crossing
| 4.  | ![alt text][test-img-4] | prob=0.73<br>prob=0.25<br>prob=0.03 | Speed limit (80km/h)<br>Speed limit (50km/h)<br>Speed limit (60km/h) |
| 5.  | ![alt text][test-img-5] | prob=0.92<br>prob=0.04<br>prob=0.02 | Speed limit (70km/h)<br>Roundabout mandatory<br>General caution<br>
| 6.  | ![alt text][test-img-6] | prob=0.99<br>prob=0.01<br>prob=0.00 | Road work<br>Children crossing<br>Right-of-way at the next intersection |
| 7.  | ![alt text][test-img-7] | prob=0.60<br>prob=0.19<br>prob=0.15 | Speed limit (100km/h)<br>Vehicles over 3.5 metric tons prohibited<br>End of speed limit (80km/h) |
| 8.  | ![alt text][test-img-8] | prob=1.00<br>prob=0.00<br>prob=0.00 | Stop<br>Speed limit (30km/h)<br>Speed limit (60km/h) |
| 9.  | ![alt text][test-img-9] | prob=1.00<br>prob=0.00<br>prob=0.00 | Yield<br>Priority road<br>No vehicles |
| 10. | ![alt text][test-img-10] | prob=0.89<br>prob=0.11<br>prob=0.00 | Speed limit (50km/h)<br>Speed limit (80km/h)<br>Speed limit (30km/h) |

Some explanation of the results:<br>
1. This is actually traffic sign forbidding parking and stopping the car. This sign was not included in training dataset. Therefore model could not recognize it correctly. It's unknown sign for the model.<br>
2. Perfect classification. No comment needed.<br>
3. Perfect classification. No comment needed.<br>
4. Again, this traffic sign is not included to training dataset. Model could not classify correctly. This is just similar to a speed limit sign.<br>
5. Again here, there is no such a traffic sign in training dataset. However model classified this very closely to speed limit to 70.<br>
6. Perfect classification. No comment needed.<br>
7. This traffic sign should be classified as _Vehicles over 3.5 metric tons prohibited_. It was classified on 2nd position with probability 19%. Problably this mistake is caused by very little amount of pictures for this class in training dataset.<br>
8. Perfect classification. No comment needed.<br>
9. Perfect classification. No comment needed.<br>
10. Very good classification with quite high probability (89%).<br>

## Potential improvements

Following further steps might improve model:<br>
* Add batch normalization in the neural network. This step usually significantly improve model performance.
* Use another activation function, that generates better results. As written in [This arxiv paper](https://arxiv.org/abs/1908.08681v1), Mish: A Self Regularized Non-Monotonic Neural Activation Function provides better results.<br>
* Improve dataset. Training dataset clearly shows lots of blurry images (e.g. in speed limit traffic signs) where the numbers are even difficult to be read by human, as well as too dark images or too exposed images, where simple "lightness correction" normalization does not bring good results. 
