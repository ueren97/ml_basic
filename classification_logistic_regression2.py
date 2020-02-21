import numpy as np
import matplotlib.pyplot as plt

# read training data
train = np.loadtxt('sample4.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

# initialize the parameters
theta = np.random.rand(4)

# standardization
mu = train_x.mean(axis=0)
sigma = train_x.std(axis=0)


def standardize(x):
    return (x - mu) / sigma


train_z = standardize(train_x)


# add x0&x3
def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    x3 = x[:, 0, np.newaxis] ** 2
    return np.hstack([x0, x, x3])


X = to_matrix(train_z)


# sigmoid function
def f(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))


# estimated time of arrival
ETA = 1e-3

# repeat
epoch = 5000


# repeat training
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)


# plot data
x1 = np.linspace(-2, 2, 100)
x2 = -(theta[0] + theta[1] * x1 + theta[3] * x1 ** 2) / theta[2]

plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 0], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x1, x2, linestyle='dashed')
plt.show()
