import numpy as np
import matplotlib.pyplot as plt

# read training data
train = np.loadtxt('sample.csv', delimiter=',', skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]


# standardization
mu = train_x.mean()
sigma = train_x.std()


def standardize(x):
    return (x - mu) / sigma


train_z = standardize(train_x)


# objective function
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)


# estimated time of arrival(ETA)
ETA = 1e-3


# create matrix of training data
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T


X = to_matrix(train_z)


# prediction function
def f(x):
    return np.dot(x, theta)


# mean square error
def MSE(x, y):
    return (1 / x.shape[0]) * np.sum((y - f(x)) ** 2)


# initialize the papameter randomly
theta = np.random.rand(3)


# the history of mean square error
errors = []

# difference of error
diff = 1

# repeat training data
errors.append(MSE(X, train_y))
while diff > 1e-2:
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    errors.append(MSE(X, train_y))
    diff = errors[-2] - errors[-1]


# plotting difference
x = np.arange(len(errors))

plt.plot(x, errors)
plt.show()
