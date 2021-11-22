# !/user/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt

# read video

import scipy.io
movie = scipy.io.loadmat('escalator_data.mat')
#frame0 =
print(np.shape(movie['X'][:, 0]))

plt.imshow(movie['X'][:, 0].reshape((160, 130)).swapaxes(0, 1), cmap='gray')
plt.show()
