#!/usr/bin/env python3

from torch.utils.data.dataset import Dataset # For custom datasets
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import os
import random
import cv2
from PIL import Image


image_directory = "/home/joshuastern/Documents/601/DeepLearning/downloads"
NUM_IMAGES = 100
CATEGORIES = ["daisy", "rose"]

class CustomTrainingDataset(Dataset):

    def __init__(self, path, NUM_IMAGES, CATEGORIES):

        self.to_tensor = transforms.ToTensor()

        training_data = []
        validation_data = []
        testing_data = []

        self.TRAINING_SIZE = NUM_IMAGES * 0.8
        self.VALIDATION_SIZE = NUM_IMAGES * 0.1
        self.TESTING_SIZE = NUM_IMAGES * 0.1

        image_directory = path
        IMG_SIZE = 256

        for category in CATEGORIES:
            counter = -1
            path = os.path.join(image_directory, category)
            classnum = 1
            if category == "daisy":
                classnum =  0
            for img in os.listdir(path):
                counter = counter + 1
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    if counter<self.TRAINING_SIZE:
                        training_data.append([new_array, classnum])
                    elif counter<(self.TRAINING_SIZE+self.VALIDATION_SIZE):
                        validation_data.append([new_array, classnum])
                    else:
                        testing_data.append([new_array, classnum])
                except Exception as e:
                    print(img, "is an invalid image and will be ignored")

        random.shuffle(training_data)
        random.shuffle(validation_data)
        random.shuffle(testing_data)

        self.train_images = []
        self.train_labels = []
        self.validation_images = []
        self.validation_labels = []
        self.test_images = []
        self.test_labels = []

        for features, label in training_data:
            self.train_images.append(features)
            self.train_labels.append(label)
        for features, label in validation_data:
            self.validation_images.append(features)
            self.validation_labels.append(label)
        for features, label in testing_data:
            self.test_images.append(features)
            self.test_labels.append(label)

        self.train_images = np.array(self.train_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        self.train_labels = np.array(self.train_labels)
        self.validation_images = np.array(self.validation_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        self.validation_labels = np.array(self.validation_labels)
        self.test_images = np.array(self.test_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        self.test_labels = np.array(self.test_labels)


    def __getitem__(self, index):

        img_as_img = self.train_images[index]
        single_image_label = self.train_labels[index]
        img_as_tensor = self.to_tensor(img_as_img)
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return int(self.TRAINING_SIZE)

class CustomTestingDataset(Dataset):

    def __init__(self, path, NUM_IMAGES, CATEGORIES):

        self.to_tensor = transforms.ToTensor()

        training_data = []
        validation_data = []
        testing_data = []

        self.TRAINING_SIZE = NUM_IMAGES * 0.8
        self.VALIDATION_SIZE = NUM_IMAGES * 0.1
        self.TESTING_SIZE = NUM_IMAGES * 0.1

        image_directory = path
        IMG_SIZE = 256

        for category in CATEGORIES:
            counter = -1
            path = os.path.join(image_directory, category)
            classnum = 1
            if category == "daisy":
                classnum =  0
            for img in os.listdir(path):
                counter = counter + 1
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    if counter<self.TRAINING_SIZE:
                        training_data.append([new_array, classnum])
                    elif counter<(self.TRAINING_SIZE+self.VALIDATION_SIZE):
                        validation_data.append([new_array, classnum])
                    else:
                        testing_data.append([new_array, classnum])
                except Exception as e:
                    print(img, "is an invalid image and will be ignored")

        random.shuffle(training_data)
        random.shuffle(validation_data)
        random.shuffle(testing_data)

        self.train_images = []
        self.train_labels = []
        self.validation_images = []
        self.validation_labels = []
        self.test_images = []
        self.test_labels = []

        for features, label in training_data:
            self.train_images.append(features)
            self.train_labels.append(label)
        for features, label in validation_data:
            self.validation_images.append(features)
            self.validation_labels.append(label)
        for features, label in testing_data:
            self.test_images.append(features)
            self.test_labels.append(label)

        self.train_images = np.array(self.train_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        self.train_labels = np.array(self.train_labels)
        self.validation_images = np.array(self.validation_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        self.validation_labels = np.array(self.validation_labels)
        self.test_images = np.array(self.test_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        self.test_labels = np.array(self.test_labels)

    def __getitem__(self, index):

        img_as_img = self.test_images[index]
        single_image_label = self.test_labels[index]
        img_as_tensor = self.to_tensor(img_as_img)
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return int(self.TESTING_SIZE)


# neural network parameters
input_size = 65536
hidden_size = 500
num_classes = 2
num_epochs = 15
batch_size = 10
learning_rate = 0.001
device = 'cpu'

#create custome datasets, pass in image directory, number of images in each category, and the names of the categories
custom_train_set = CustomTrainingDataset(image_directory, NUM_IMAGES, CATEGORIES)
custom_test_set = CustomTestingDataset(image_directory, NUM_IMAGES, CATEGORIES)

# Data loader is used by pytorch and passed to the deep learning model
train_loader = torch.utils.data.DataLoader(dataset=custom_train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=custom_test_set, batch_size=batch_size, shuffle=False)

#define sequentail neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)


#specify how to train the model (what optimizer, loss type)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#start the training of the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        try:
            # reshape images to the same size
            images = images.reshape(-1, 256*256).to(device)
            labels = labels.to(device)

            # pass the data through the model
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        except:
            print("Error in training set, check that dataset is made of valid images and pytorch libraries are imported")

#compare how the model performs on the test data
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        try:
            images = images.reshape(-1, 256*256).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        except:
            print("Error in testing set, check that dataset is made of valid images and pytorch libraries are imported")

    print('Test accuracy: {} %'.format(100 * correct / total))
