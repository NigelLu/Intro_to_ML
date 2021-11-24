# !/user/bin/env python    
# -*- coding:utf-8 -*-
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob

# img1 = mpimg.imread('1/00001_00000_00012.png')
# plt.subplot(141)
# plt.imshow(img1)
# plt.axis('off')
# plt.subplot(142)
# img2 = mpimg.imread('2/00014_00001_00019.png')
# plt.imshow(img2)
# plt.axis('off')
# plt.subplot(143)
# img3 = mpimg.imread('3/00035_00008_00023.png')
# plt.imshow(img3)
# plt.axis('off')
# plt.subplot(144)
# img4 = mpimg.imread('4/00039_00000_00029.png')
# plt.imshow(img4)
# plt.axis('off')
# plt.show()
#
# # resize image example
# img1 = cv2.resize(img1, (75, 75))
# img2 = cv2.resize(img2, (75, 75))
# img3 = cv2.resize(img3, (75, 75))
# img4 = cv2.resize(img4, (75, 75))

# reading data
image_size = (100, 100)
img_list = []
target_list = []
for folder in range(1, 5):
    folder_name = str(folder)
    for img in glob.glob(folder_name + "/*.png"):
        img_temp = cv2.imread(img)
        img_temp = cv2.resize(img_temp, image_size)
        img_list.append(img_temp)
        target_list.append(folder-1)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


img_list = np.asarray(img_list)
target_list = np.asarray(target_list).reshape(-1, 1)
img_list, target_list = unison_shuffled_copies(img_list, target_list)

train_image = np.asarray(img_list[:3500])
train_labels = np.asarray(target_list[:3500]).reshape(-1, 1)

test_image = np.asarray(img_list[3500:])
test_labels = np.asarray(target_list[3500:]).reshape(-1, 1)

model = Sequential()

model.add(Conv2D(16, (3, 3), padding='same', input_shape=(image_size[0], image_size[1], 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides=None))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2), strides=None))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2), strides=None))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))

print(model.summary())

model.add(layers.Flatten())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(4, activation="softmax"))
print(model.summary())

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer="adam",
              metrics=['accuracy'])

batch_size = 32
epochs = 30

history = model.fit(train_image, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(test_image, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
#
# print(test_acc)
plt.show()
