# EC601 Mini-Project 2: Deep Learning

This purpose of this deep learning project was to explore different machine learning frameworks and develop a model for performing image recognition between two objects (I chose roses vs daisies). The two frameworks used were Tensorflow and Pytorch, and a comparison of these two systems is provided at the end of this README.

<img align="left" width="400" height="300" src="https://github.com/joshstern1/DeepLearning-601/blob/master/96.%20single-rose.jpg">
<img align="right" width="400" height="300" src="https://github.com/joshstern1/DeepLearning-601/blob/master/89.%20daisy-712892__340.jpg">

## Image Collection
To collect images for the dataset and tag them, I downloaded photos from Google Images using an open-source project: google-image-download. I used this repository to write dataset_download.py. This program is simple to use and understand. You just have to input keywords to search and the number of photos to download, and the script will download that number of photos from google images for each keyword provided. The photos are then separated (tagged) by being downloaded into their directories separated by keyword, making it easy to later access and label the images.

To use dataset_download.py, just input into the 'keywords' field that objects that you want your model to be able to recognize. Then input into the 'limit' field the total number of images that you want to download for each keyword.

https://github.com/joshstern1/DeepLearning-601/blob/71966f5213086d4b99be53f9145d5309d6f7035a/dataset_download.py#L9

You will also need to run the following command to install google-images-download from its repository:

pip install google_images_download

## Running Tensorflow Program

For my Tensorflow program, I used a Tensorflow keras tutorial on basic classification that can be found here:

https://www.tensorflow.org/tutorials/keras/basic_classification

In both my Tensorflow and Pytorch scripts, I built my custom dataset by reading in all the downloaded images into an array. It is easy to label these photos, because I can label image i the based off the directory that it is found in. The only changes that needs to be made to the script are in the variables 'image_directory', 'NUM_IMAGES', and 'CATEGORIES'. In both the Tensorflow and Pytroch programs, the variable 'image_directory' holds the directory holding the downloaded images and must be changed to whatever directory on your computer is holding the images. The variable 'NUM_IMAGES' should be changed to the dataset size for each category, and the 'CATEGORIES' array should be altered so that it contains the names of the sub-directories containing the downloaded images.

https://github.com/joshstern1/DeepLearning-601/blob/42382c25341030610ab56f243dcaa0dd2b2d3ea2/tensorflow_model.py#L10-L17

To run tensorflow_model.py, first perform the proper installations by running the following commands:

pip install tensorflow
pip install opencv-python

## Running Pytorch Program

For the deep learning model for my Pytorch program, I used a Pytorch tutorial that can be found here:

https://github.com/yunjey/pytorch-tutorial

Again in this program, I defined three variables, 'image directory', 'NUM_IMAGES', and 'CATEGORIES'. These 3 variables can be changed depending on the type of images you are comparing and which directory they can be found in.

https://github.com/joshstern1/DeepLearning-601/blob/b7944abc7cb51902b3e0ca176131adc75aa29a41/pytorch_model.py#L16-L18

To run pytorch_model.py, first perform the proper installations by running the following commands:

pip3 install torchvision
pip3 install pillow


## System Comparison: Tensorflow vs PyTorch

### Tensorflow:

Tensorflow is the most popular library used for deep learning. It was developed by Google and can be used to develop neural networks for image recognition, which is what it was used for in this project. Tensorflow comes with plenty of documentation and tutorials, which ease the learning curve for beginners. Within Tensorflow, we use Keras, a high-level API for building neural networks. In comparing the process of developing deep learning models in Tensorflow with and without Keras, I found that Keras greatly simplifies computation required for neural nets. Instead of writing code for each multiplication and addition that needs to take place, Keras allows you to create an entire model, define optimizers, and start the learning process in just a few lines of code. Keras additionally simplified the process for testing the network with the testing dataset by reducing it to a single command.

### Pytorch:

The other deep learning framework used was PyTorch, which is newer than Tensorflow but rising quickly in popularity. Like Tensorflow, Pytorch abstract the neural network by reducing it to just a few lines of code. There is less community support and tutorial for Pytorch, making it more difficult to get started. However, after a few hours of studying the framework, I found it relatively easy to build the model. Pytorch comes with a lot of support for using pre-trained models and existing datasets, but it is lacking in its documentation for custom datasets. The Pytorch dataloader simplified the building of the model, but introduced minor complexities into getting the dataset working. Pytorch also required that image data be converted into tensors, which helps in widening Pytorch's support to GPUs.
