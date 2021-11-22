# !/user/bin/env python    
# -*- coding:utf-8 -*-
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPooling2D


model = Sequential()

# model.add(Conv2D(num_units, (filter_size1, filter_size2), padding='same',
#                              input_shape=(3, IMG_SIZE, IMG_SIZE),
#                              activation='relu'))

