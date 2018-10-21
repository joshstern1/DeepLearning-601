# EC601 Mini-Project 2: Deep Learning

This purpose of this deep learning project was to explore different machine learning frameworks and develop a model for image recognition. The two frameworks used were Tensorflow and Pytorch, and a comparison of these two systems is provided at the end of this README.

I used these frameworks to develop a model for image recognition between roses vs daisies. To collect images for the dataset and tag them, I downloaded photos from Google Images using an open-source project: google-image-download. I used this repository to write image_download.py. This program simple to understand and use. One just has to input keywords to search and the number of photos to download, and the script will download that number of photos from google images based off the keywords provided. The photos are then separated by being downloaded into their own directories, making it easy to later access.

To use image-download.py, just input into the 'keywords' field that objects that you want your model to be able to recognize. The input into the 'limit' field the total number of images that you want to download. You will also need to run the following command to install google-images-download from its repository:

pip install google_images_download


In both my Tensorflow and Pytorch scripts, I build my custom dataset by reading in all the downloaded images into an array. It is easy to label these photos, because I know what the image is of based off the directory that it is found in. The only change that needs to be made to the script is the directory containing the photos downloaded by image-download.py. In both programs, the variable 'DATADIR' holds this directory and must be changed to whatever directory on your computer is holding the images.

To run tensorflow_model.py, first perform the proper installations by running the following commands:

pip install tensorflow
pip install opencv-python

To run pytorch_model.py, first perform the proper installations by running the following commands:
pip3 install torchvision
pip3 install pillow


System Comparison: Tensorflow vs PyTorch

Tensorflow:

Tensorflow is the most popular library used for deep learning. It was developed by Google and can be used to develop neural networks for image recognition, which is what it was used for in this project. Tensorflow comes with plenty of documentation and tutorials, which ease the learning curve for beginners. Within Tensorflow, we use Keras, a high-level API for building neural networks. In comparing the process of developing deep learning models in Tensorflow with and without Keras, I found that Keras greatly simplifies computation required for neural nets. Instead of writing code for each multiplication and addition that needs to take place, Keras allows you to create an entire model, define optimizers, and start the learning process in just a few lines of code. Keras additionally simplified the process for testing the network with the testing dataset by reducing it to a single command.

Pytorch:

The other deep learning framework used was PyTorch, which is newer than Tensorflow but rising quickly in popularity. Like Tensorflow, Pytorch abstract the neural network by reducing it to just a few lines of code. There is less community support and tutorial for Pytorch, making it more difficult to get started. However, after a few hours of studying the framework, I found it relatively easy to build the model. Pytorch comes with a lot of support for using pre-trained models and existing datasets, but it is lacking in its documentation for custom datasets. The Pytorch dataloader simplified the building of the model, but introduced minor complexities into getting the dataset working. Pytorch also required that image data be converted into tensors, which helps in widening Pytorch's support to GPUs.
