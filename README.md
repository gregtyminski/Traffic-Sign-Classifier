# Project: Build a Traffic Sign Recognition Program

[//]: # (Image References)
[step-1-lenet-best]: ./images/step-1-lenet-best.png "LeNet - best results"
[step-1-lenet-t2]: ./images/step-1-lenet-t2.png "LeNet - graph with best results"

This project is a part of:  
 [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

In this project we are going to train a network to recognize traffic signs.\
The dataset of traffic signs come from [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) and this dataset will be used to train the neural network recognizing traffic signs.\
This dataset is to large to be kept in GitHub, therefore if you want to run any training job please download the dataset from [this link](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip) and unpack it to `data` directory.\
\
To track results of all training experiments I use [Neptune.ml](http://neptune.ml) tool.
## Step 0: Load The Data

At the beginning we need to load the dataset. Dataset is kept in the folder `data` and it contains following files:\
```bash
total 311760
-rw-r--r--@ 1 grzegorz.tyminski  staff   38888118 Nov  7  2016 test.p
-rw-r--r--@ 1 grzegorz.tyminski  staff  107146452 Feb  2  2017 train.p
-rw-r--r--@ 1 grzegorz.tyminski  staff   13578712 Feb  2  2017 valid.p
```
Class loading this dataset is implemented in the file [traffic_sign_dataset.py](traffic_sign_dataset.py).\
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
The class `TrafficData` contains as well method `normalize_data()` to normalize the dataset (i.e. change values of pixels from 0..255 to 0..1) as well as the method `shuffle_dataset()` to randomize the order of images in dataset.
 
## Step 1: Train same LeNet as in MNIST example 

In the very first step we are going to train the same LeNet neural network on as is the MNIST example.\
Let's just verify, how good it is.\
Class `LeNet` with model architecture is defined in the [LeNet.py](LeNet.py) file.\
It's input and output shape is adjusted for Traffic Sign Dataset (3-channel color with 43-classes output).\
I've run grid search over hyperparams with values for `epoch` _(10 or 15)_, `batch_size` _(32, 64, 128, 256, 512, 1024)_ and `learn_rate` _(0.0005, 0.001, 0.002)_
For several trials we got the validation accuracy up to almost __0.949887__ for `epochs` equal to 10, `batch_size` equal to 64 and `learn_rate` equal to 0.002\
![alt text][step-1-lenet-best]
The results did not differ much for several hyperparams. However we can clearly show, that smaller `batch_size` resulted in better result. Values for `learn_rate` below _0.001_ did not give satisfying results and the number of `epochs` bigger than 10 gave no better results (see graph below).
![alt text][step-1-lenet-t2]