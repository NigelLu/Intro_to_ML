# !/user/bin/env python
# -*- coding:utf-8 -*-
from __future__ import division

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import fbpca

# read video

import scipy.io
movie = scipy.io.loadmat('escalator_data.mat')
#frame0 =
print(np.shape(movie['X'][:, 0]))

plt.imshow(movie['X'][:, 0].reshape((160, 130)).swapaxes(0, 1), cmap='gray')
plt.show()


def robustPCA(X, delta=1e-6, mu=None, maxiter=10):
	"""
	The function should return a PCA like part stored in 'L' with only a few singular values
	that are non zero and a sparse sequence 'S' in which the images are black except w very
	limited number of pixels
	"""

	# Initialize the tuning parameters.
	lam = 1 / max(X.shape[0], X.shape[1])
	if mu is None:
		# complete with your value for mu
		mu = X.shape[0] * X.shape[1] / (4 * np.linalg.norm(X, ord=1, axis=None))

	# Convergence criterion.
	norm = np.sum(X ** 2)

	# Iterate.
	i = 0
	# rank = np.min(X.shape[0], X.shape[1])
	S = np.zeros((X.shape[0], X.shape[1]))
	Y = np.zeros((X.shape[0], X.shape[1]))
	while i < max(maxiter, 1):

		# Step 1. Compute and truncate the SVD
		step1_temp = X - S - (1/mu) * Y
		u, sigma_arrary, vt = fbpca.pca(step1_temp, raw=True)
		truncated_sigma = np.sign(sigma_arrary) * np.maximum(np.abs(sigma_arrary)-mu, 0)
		truncated_sigma_matrix = np.zeros((u.shape[1], vt.shape[0]))
		for j in range(len(truncated_sigma)):
			truncated_sigma_matrix[j, j] = truncated_sigma[j]
		L = np.matmul(np.matmul(u, truncated_sigma_matrix), vt)

		# Step 2. Truncate the entries of X - L + mu^(-1)Y
		step2_temp = X - L + (1/mu) * Y
		S = np.sign(step2_temp) * np.maximum(step2_temp - lam*mu, 0)

		# Step 3. Update the matrix Y
		Y = Y + mu * (X - L - S)

		# Convergence criterion
		err = np.sqrt(np.sum((X-L-S) ** 2) / norm)

		print(np.sum((X-L-S)**2))
		if err < delta:
			break
		i += 1

	return L, S


L_result, S_result = robustPCA(movie["X"])
plt.imshow(L_result[:, 0].reshape((160, 130)).swapaxes(0, 1), cmap='gray')
plt.show()
