# !/user/bin/env python    
# -*- coding:utf-8 -*-
from tensorflow.keras.layers import Conv2D

from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import MaxPooling2D
import matplotlib.image as mpimg
import cv2

img1 = mpimg.imread('1/00001_00000_00012.png')
plt.subplot(141)
plt.imshow(img1)
plt.axis('off')
plt.subplot(142)
img2 = mpimg.imread('2/00014_00001_00019.png')
plt.imshow(img2)
plt.axis('off')
plt.subplot(143)
img3 = mpimg.imread('3/00035_00008_00023.png')
plt.imshow(img3)
plt.axis('off')
plt.subplot(144)
img4 = mpimg.imread('4/00039_00000_00029.png')
plt.imshow(img4)
plt.axis('off')
plt.show()

# resize images
img1 = cv2.resize(img1, (75, 75))
img2 = cv2.resize(img2, (75, 75))
img3 = cv2.resize(img3, (75, 75))
img4 = cv2.resize(img4, (75, 75))

model = Sequential()

# model.add(Conv2D(num_units, (filter_size1, filter_size2), padding='same',
#                              input_shape=(3, IMG_SIZE, IMG_SIZE),
#                              activation='relu'))

