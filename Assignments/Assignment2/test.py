# !/user/bin/env python    
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def FISTA(X, t, eta1, beta0, lbda):
    '''function should return the solution to the minimization of the LASSO objective ||X*beta - t||_2^2 + lambda*||beta||_1
    by means of FISTA updates'''
    # set up variables for iterations
    MaxIter = 10000
    loss_fista = np.zeros(MaxIter)
    beta = beta0
    beta_prev = beta0
    eta = eta1
    eta_prev = eta1
    y = beta0 # y1=beta0
    N = np.shape(X)[0] # size of dataset
    loss_ista = np.zeros((MaxIter, 1))
    # normalize features X and target values t
    X = X - np.ones((N,N))@X/N
    t = t - np.ones((N,N))@t/N
    t = t.reshape(-1,1)
    temp_prev = 1
    temp = 1
    # fista iterations
    for k in range(1, MaxIter): # k starts from 1
        # the first line, updating beta using a single ISTA update
        arg = y - 2*eta*X.transpose() @ (X @ y - t)
        # print('arg=',arg) for some reason it is blowing up
        beta = np.maximum(np.absolute(arg)-lbda*eta, 0.) * np.sign(arg)
    # the second line, updating eta
        temp = (1 + np.sqrt(1+4*temp_prev**2)) / 2
        # the third line, updating y
        y = beta + (temp_prev-1)/temp * (beta-beta_prev)
        # update for beta_prev and eta_prev
        beta_prev = beta
        temp_prev = temp
        # record the loss
        loss_fista[k] = np.sum((X @ beta - t) ** 2) + lbda*np.sum(beta)
    print(loss_ista)
    return beta , loss_fista # beta is the final beta_LASSO


# set up the true model
x = np.linspace(0,1,20)
x_true = np.linspace(0,1,100)
t_true = 0.1 + 1.3*x_true
# set up our data points, given by the true model with gaussian perturbation
t_noisy = 0.1 + 1.3*x + np.random.normal(0,0.1,len(x))
plt.scatter(x, t_noisy, c='r')
plt.plot(x_true, t_true)
# set up the regularization parameter lambda
lbda = 0.1
# try polynomials with different degree d
degree = [2,3,4,5,6,7,8,9]
for d in degree:
    beta0 = np.random.normal(0,0.1,size=d).reshape(-1,1) # randomly initialize beta
    poly = PolynomialFeatures(d) # initialize the polynomail model
    X_poly = poly.fit_transform(x.reshape((-1,1)))[:,1:] # tranform x into polynomial features and discard the "dummy" feature x^0=1
    # run the LASSO update using FISTA
    beta_LASSO, loss_fista = FISTA(X_poly, t_noisy, 1, beta0, lbda)
    # to make predictions and display the result, we need to first centralize the input data
    N = np.shape(X_poly)[0] # number of data points
    X_polyMean = np.ones((N,N))@X_poly/N # this is what we subtracted from the features for centralization
    t_noisyMean = np.ones((N,N))@t_noisy/N # this is what we subtracted from the targets for centralization
    plt.plot(x , (X_poly-X_polyMean) @ beta_LASSO + t_noisyMean, linewidth=0.5 )
    print('d=%d, beta ='%d, beta_LASSO.flatten())
plt.legend(["d=%d"%d for d in degree])
plt.show()
