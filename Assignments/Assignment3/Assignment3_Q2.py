import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

pointsClass1 = loadmat('KernelPointsEx4class1.mat')['PointsEx4class1']
pointsClass2 = loadmat('KernelPointsEx4class2.mat')['PointsEx4class2']


plt.scatter(pointsClass1[:, 0], pointsClass1[:, 1], c='r')
plt.scatter(pointsClass2[:, 0], pointsClass2[:, 1], c='b')
plt.show()


# I modified this function below because I believe we need more inputs
# so as to compute the hinge loss along with its gradient
def HingeLoss(x, beta_vec, beta_0):
    """Returns the value and gradient of the hinge
    loss at the point x
    if the gradient does not exist, then we return the gradient as None (i.e. if hinge loss = 0)"""

    value = None
    gradient = None

    t_i = x[-1]
    x_i = np.asarray((x[:-1]))

    # first compute the hinge loss
    value = max(0, 1 - t_i * (np.matmul(beta_vec, x_i.reshape(-1, 1)) + beta_0))
    if isinstance(value, np.ndarray):
        value = value[0]

    # compute the gradient
    if value == 0:
        # in this case, the gradient does not exist
        return value, gradient
    gradient = -t_i * x[0:2]
    return value, gradient


def HingeLossSVC(beta_init, beta0_init, training):
    """Returns the maximal margin classifier for the
    training dataset"""
    num_iter = 1000
    eta = 0.05
    C = 0.075
    num_sample = training.shape[0]
    hinge_loss_list = []
    beta = beta_init
    beta0 = beta0_init

    for _ in range(num_iter):
        gradient_temp = np.zeros((beta.shape[0], ))
        beta0_gradient = 0
        hinge_loss_temp = 0

        for i in range(num_sample):
            hinge_loss_step, gradient_step = HingeLoss(training[i], beta, beta0)
            if hinge_loss_step != 0:
                beta0_gradient += -training[i][-1]
                gradient_temp += gradient_step
            hinge_loss_temp += hinge_loss_step

        hinge_loss_list.append(hinge_loss_temp)
        gradient_temp *= C/num_sample
        gradient_temp += 2 * beta
        beta0 -= eta * beta0_gradient * C/num_sample
        beta -= eta * gradient_temp

    return beta, beta0, hinge_loss_list


pointsClass1_labeled = np.hstack((pointsClass1, np.ones((pointsClass1.shape[0], 1))))
pointsClass2_labeled = np.hstack((pointsClass2, -np.ones((pointsClass2.shape[0], 1))))
training_set = np.vstack((pointsClass1_labeled, pointsClass2_labeled))

beta, beta0, hinge_loss_list = HingeLossSVC(np.ones((2,)), 0, training_set)
y_coefficient = -beta[0]/beta[1]
x_list = np.linspace(0, 1, 100)
y_list = x_list * y_coefficient + beta0

hinge_loss_list = np.asarray(hinge_loss_list)

plt.plot(x_list, y_list, c="y")
plt.scatter(pointsClass1[:, 0], pointsClass1[:, 1], c='r')
plt.scatter(pointsClass2[:, 0], pointsClass2[:, 1], c='b')
plt.show()
