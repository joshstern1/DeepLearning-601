#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2
import random

#change image_directory to whatever directory holds the downloaded Images
image_directory = "/home/joshuastern/Documents/601/DeepLearning/downloads"

#change the contents of CATEGORIES to the two categories that you want your model to recognize
CATEGORIES = ["daisy", "rose"]

#change NUM_IMAGES to the total amount of images in your dataset for each category
NUM_IMAGES = 100
TRAINING_SIZE = NUM_IMAGES * 0.8    #training set is 80% of your data
VALIDATION_SIZE = NUM_IMAGES * 0.1  #validation set is 10% of your data
TESTING_SIZE = NUM_IMAGES * 0.1     #testing set is 10% of your data

IMG_SIZE = 256
training_data = []
validation_data = []
testing_data = []

#read in images from directory into an array
for category in CATEGORIES:
    counter = -1
    path = os.path.join(image_directory, category)
    classnum = 1
    if category == CATEGORIES[0]:
        classnum =  0
    for img in os.listdir(path):
        counter = counter + 1
        try:
            #use opencv to open and format images
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            if counter<TRAINING_SIZE:
                #build training data array
                training_data.append([new_array, classnum])
            elif counter<(TRAINING_SIZE+VALIDATION_SIZE):
                #build validation data array
                validation_data.append([new_array, classnum])
            else:
                #build testing data array
                testing_data.append([new_array, classnum])
        except Exception as e:
            print(img, "is an invalid image and will be ignored")

#shuffle the data so that the daisies are mixed in with the roses
random.shuffle(training_data)
random.shuffle(validation_data)
random.shuffle(testing_data)

train_images = []
train_labels = []
validation_images = []
validation_labels = []
test_images = []
test_labels = []

#build separate arrays for the images and their respective labels
for features, label in training_data:
    train_images.append(features)
    train_labels.append(label)
for features, label in validation_data:
    validation_images.append(features)
    validation_labels.append(label)
for features, label in testing_data:
    test_images.append(features)
    test_labels.append(label)

#change the image and label arrays into numpy arrays for the keras model
train_images = np.array(train_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
train_labels = np.array(train_labels)
validation_images = np.array(validation_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
validation_labels = np.array(validation_labels)
test_images = np.array(test_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_labels = np.array(test_labels)

#normalize the arrays so that they fit the expected input of the keras model
train_images = tf.keras.utils.normalize(train_images, axis=1)
validation_images = tf.keras.utils.normalize(validation_images, axis=1)
test_images = tf.keras.utils.normalize(test_images, axis=1)

#Building and added hidden layers to sequential neural network
#Can add and remove layers to your liking
model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

#specify how to train the model (what optimizer, loss type)
#can change the loss model to other types such as binary_crossentropy
try:
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
except:
    print("error specifying model optimizer and loss type")

#start the training of the model
#can change the epochs argument to control the number of times that training data is passed through the model
try:
    model.fit(train_images, train_labels, epochs=20, validation_data = (validation_images, validation_labels))
except:
    print("error occured while training model")

#compare how the model performs on the test data
try:
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
except:
    print("error occured testing the model")
