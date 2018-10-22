#!/usr/bin/env python3

from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

#for the keywords, replace with whatever type of objects you want the deep learning model to recognize
#the limit controls the number of photos that are downlaoded from google images. 
#In order to download over 100 photos, you must have google chrome installed
arguments = {"keywords":"rose,daisy","limit":100,"print_urls":True}
paths = response.download(arguments)
  
