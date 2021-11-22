# !/user/bin/env python    
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


max_iter = 1000

"""Question 1.1"""
def ISTA(beta_init, X, t, lbda, eta):
	'''The function takes as input an initial guess for beta, a set
	of feature vectors stored in X and their corresponding
	targets stored in t, a regularization weight lbda,
	step size parameter eta and must return the
	regression weights following from the minimization of
	the LASSO objective'''

	# "center the data"
	# X = np.copy(X)
	# for column_position in np.arange(np.shape(X)[1]):
	# 	X[:, column_position] = X[:, column_position] - np.average(X[:, column_position])
	#
	# t[:, 0] = t[:, 0] - np.average(t[:, 0])

	ISTA_loss_temp = np.zeros((max_iter, 1))
	beta_LASSO = np.zeros((np.shape(X)[1], 1))

	# add your code here
	beta_LASSO += beta_init
	for i in np.arange(max_iter):
		descent_temp1 = beta_LASSO - 2 * eta * np.matmul(X.transpose(), (np.matmul(X, beta_LASSO) - t))
		# beta_LASSO = np.multiply(np.maximum((np.absolute(descent_temp1) - lbda * eta), 0.), np.sign(descent_temp1))
		beta_LASSO = np.sign(descent_temp1) * np.maximum(np.abs(descent_temp1) - lbda*eta, 0.)
		ISTA_loss_temp[i, 0] = np.sum((np.matmul(X, beta_LASSO) - t)**2)/np.shape(X)[0]

	# "de-center beta_0"
	# beta_LASSO[:, 0] *= np.shape(t)[0]
	return beta_LASSO, ISTA_loss_temp


"""Question 1.2"""
from sklearn.preprocessing import PolynomialFeatures


x = np.linspace(0, 1, 20)
xtrue = np.linspace(0, 1, 100)
t_true = 0.1 + 1.3 * xtrue

t = 0.1 + 1.3 * x

tnoisy = t + np.random.normal(0, .1, len(x))

plt.scatter(x, tnoisy, c='r')
plt.plot(xtrue, t_true, c="g")
plt.show()


eta = 0.001
lbda = 0.05

beta_init = np.hstack((np.asarray([-20]), np.random.random(1)*50)).reshape(2, 1)
XPrediction = np.hstack((np.ones((len(x),)).reshape(-1, 1), x.reshape(-1, 1)))
beta_LASSO_ISTA, ISTA_loss = ISTA(beta_init, XPrediction, tnoisy.reshape(-1, 1), lbda, eta)
tPrediction = np.matmul(XPrediction, beta_LASSO_ISTA)

"""Poly degree 8"""
poly_8 = PolynomialFeatures(8)
XPrediction_poly8 = poly_8.fit_transform(XPrediction)
beta_LASSO_ISTA_8, ISTA_loss_8 = ISTA(np.ones((45, 1)), XPrediction_poly8, tnoisy.reshape(-1, 1), lbda, eta)
tPrediction_8 = np.matmul(XPrediction_poly8, beta_LASSO_ISTA_8)

"""Poly degree 9"""
poly_9 = PolynomialFeatures(9)
XPrediction_poly9 = poly_9.fit_transform(XPrediction)
beta_LASSO_ISTA_9, ISTA_loss_9 = ISTA(np.ones((55, 1)), XPrediction_poly9, tnoisy.reshape(-1, 1), lbda, eta)
tPrediction_9 = np.matmul(XPrediction_poly8, beta_LASSO_ISTA_8)

"""Poly degree 10"""
poly_10 = PolynomialFeatures(10)
XPrediction_poly10 = poly_10.fit_transform(XPrediction)
beta_LASSO_ISTA_10, ISTA_loss_10 = ISTA(np.ones((66, 1)), XPrediction_poly10, tnoisy.reshape(-1, 1), lbda, eta)
tPrediction_10 = np.matmul(XPrediction_poly8, beta_LASSO_ISTA_8)


plt.plot(xtrue, t_true, c="g", label="True")
plt.plot(x, tPrediction, c="r", label="Degree 2")
plt.scatter(x, tnoisy, c='r')
plt.legend()
plt.show()

plt.plot(x, tPrediction_8, c="y", label="Degree 8")
plt.scatter(x, tnoisy, c='r')
plt.legend()
plt.show()

plt.plot(x, tPrediction_9, c="b", label="Degree 9")
plt.scatter(x, tnoisy, c='r')
plt.legend()
plt.show()

plt.plot(x, tPrediction_10, c="g", label="Degree 10")
plt.scatter(x, tnoisy, c='r')
plt.legend()
plt.show()

print(beta_LASSO_ISTA)


"""Question 1.3"""


def FISTA(X, t, eta0, beta0, lbda):
	'''function should return the solution to the minimization of the
	the LASSO objective ||X*beta - t||_2^2 + lambda*||beta||_1
	by means of FISTA updates'''

	eta = eta0
	y_LASSO = beta0
	t_temp = 1
	FISTA_loss_temp = np.zeros((max_iter, 1))

	for i in np.arange(max_iter):
		descent_temp = y_LASSO - 2 * eta * np.matmul(X.transpose(), (np.matmul(X, y_LASSO) - t))
		x_temp = np.sign(descent_temp) * np.maximum(np.abs(descent_temp) - lbda*eta, 0.)
		t_new = (1+np.sqrt(1+4*t_temp**2))/2
		y_LASSO = x_temp + ((t_temp-1)/t_new)*(x_temp-y_LASSO)
		t_temp = t_new
		FISTA_loss_temp[i, 0] = np.sum((np.matmul(X, y_LASSO) - t) ** 2)/np.shape(X)[0]

	return y_LASSO, FISTA_loss_temp


beta_LASSO_FISTA, FISTA_loss = FISTA(XPrediction, tnoisy.reshape(-1, 1), eta, beta_init, lbda)

tPrediction = np.matmul(XPrediction, beta_LASSO_FISTA)

plt.plot(xtrue, t_true, c="g", label="True")
plt.plot(x, tPrediction, c="r", label="FISTA")
plt.scatter(x, tnoisy, c='r')
plt.legend()
plt.show()
print(beta_LASSO_FISTA)


"""Question 1.4"""
plt.plot(FISTA_loss, c="b")
plt.plot(ISTA_loss, c="r")
plt.show()


"""Question 2.1"""
import scipy.io
from matplotlib.colors import ListedColormap
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

class1 = scipy.io.loadmat('points_Assignment2_Ex2_class1.mat')['points_Assignment2_Ex2_class1']
class2 = scipy.io.loadmat('points_Assignment2_Ex2_class2.mat')['points_Assignment2_Ex2_class2']
class3 = scipy.io.loadmat('points_Assignment2_Ex2_class3.mat')['points_Assignment2_Ex2_class3']
class4 = scipy.io.loadmat('points_Assignment2_Ex2_class4.mat')['points_Assignment2_Ex2_class4']


plt.scatter(class1[:, 0], class1[:, 1], c="r")
plt.scatter(class2[:, 0], class2[:, 1], c="g")
plt.scatter(class3[:, 0], class3[:, 1], c="b")
plt.scatter(class4[:, 0], class4[:, 1], c="y")
plt.show()

"Construct the target dataset"
class1_with_label = np.append(class1, [[0]]*np.shape(class1)[0], axis=1)
class2_with_label = np.append(class2, [[1]]*np.shape(class2)[0], axis=1)
class3_with_label = np.append(class3, [[2]]*np.shape(class3)[0], axis=1)
class4_with_label = np.append(class4, [[3]]*np.shape(class4)[0], axis=1)

data_with_label = np.vstack((class1_with_label, class2_with_label, class3_with_label, class4_with_label))

print("123")


axis_0_guess = []
axis_1_guess = []
"""Define a function that does one split step for us
   @:param j: which dimension we wanna do split on
   @:param data_temp: the dataset we are going to split along with labels
   @:param low: the starting lower bound
   @:param high: the ending upper bound
   @:param current_loss: the loss that we got by doing the last split
   
   @:return [s]: the splitting points on axis"""
def split(j, data_temp, low, high, current_loss):
	if current_loss <= 0.1:
		return []
	step_size = 0.01
	num_iter = (high-low)//step_size

	# "sort data_temp for the given dimension j"
	# data_temp = data_temp[np.argsort(data_temp[:j])]

	loss_list = []

	for i in np.arange(num_iter):
		R1 = data_temp[(data_temp[:, j]<=i*step_size)]
		R2 = data_temp[(data_temp[:, j]>i*step_size)]
		c1_hat = np.average(R1[:, 2])
		c2_hat = np.average(R2[:, 2])

		loss_R1 = np.sum((R1[:, 2]-c1_hat)**2)
		loss_R2 = np.sum((R2[:, 2]-c2_hat)**2)

		loss_list.append(loss_R1+loss_R2)

	min_loss = min(loss_list)
	position = loss_list.index(min_loss)
	s = position*step_size + low

	R1 = data_temp[(data_temp[:, j] <= s)]
	R2 = data_temp[(data_temp[:, j] > s)]
	c1_hat = np.average(R1[:, 2])
	c2_hat = np.average(R2[:, 2])

	loss_R1 = np.sum((R1[:, 2] - c1_hat) ** 2)
	loss_R2 = np.sum((R2[:, 2] - c2_hat) ** 2)
	if low == s or high == s:
		return []
	return split(j, R1, low, s, loss_R1) + [s] + split(j, R2, s, high, loss_R2)


min_loss_axis_0 = float("inf")
min_loss_axis_1 = float("inf")
axis_0_splitpoint_list = [0]
axis_1_splitpoint_list = [0]
"Trial starting with feature 1 and then go on with feature 2"
axis_0_splitpoint_list += split(0, data_with_label, 0, 1, min_loss_axis_0)
axis_1_splitpoint_list += split(1, data_with_label, 0, 1, min_loss_axis_1)
axis_0_splitpoint_list += [1]
axis_1_splitpoint_list += [1]
plt.scatter(class1[:, 0], class1[:, 1], c="r")
plt.scatter(class2[:, 0], class2[:, 1], c="g")
plt.scatter(class3[:, 0], class3[:, 1], c="b")
plt.scatter(class4[:, 0], class4[:, 1], c="y")

for split_point in axis_0_splitpoint_list:
	plt.axvline(x=split_point)
for split_point in axis_1_splitpoint_list:
	plt.axhline(y=split_point)

for position_0, axis_0_splitpoint in enumerate(axis_0_splitpoint_list[1:]):
	for position_1, axis_1_splitpoint in enumerate(axis_1_splitpoint_list[1:]):
		region_to_consider = [(axis_0_splitpoint_list[position_0], axis_1_splitpoint_list[position_1]), (axis_0_splitpoint, axis_1_splitpoint)]
		data_to_consider = data_with_label[(data_with_label[:, 0] >= region_to_consider[0][0]) & (data_with_label[:, 0] <= region_to_consider[1][0]) &
										   (data_with_label[:, 1] >= region_to_consider[0][1]) & (data_with_label[:, 1] <= region_to_consider[1][1])]
		region_c_hat = np.average(data_to_consider[:, 2])
		region_guess = round(region_c_hat, 0)
		if region_guess == 0:
			plt.axvspan(region_to_consider[0][0], region_to_consider[1][0],
						ymin=region_to_consider[0][1], ymax=region_to_consider[1][1], color="red", alpha=0.13)
		elif region_guess == 1:
			plt.axvspan(region_to_consider[0][0], region_to_consider[1][0],
						ymin=region_to_consider[0][1], ymax=region_to_consider[1][1], color="green", alpha=0.13)
		elif region_guess == 2:
			plt.axvspan(region_to_consider[0][0], region_to_consider[1][0],
						ymin=region_to_consider[0][1], ymax=region_to_consider[1][1], color="blue", alpha=0.13)
		elif region_guess == 3:
			plt.axvspan(region_to_consider[0][0], region_to_consider[1][0],
						ymin=region_to_consider[0][1], ymax=region_to_consider[1][1], color="yellow", alpha=0.13)
plt.show()





