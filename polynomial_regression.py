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

# initialize the parameter
theta = np.random.rand(3)


# create matrix of training data
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]), x, x ** 2]).T


X = to_matrix(train_z)


# prediction function
def f(x):
    return np.dot(x, theta)


# difference of error
diff = 1

# repeat learning
error = E(X, train_y)
while diff > 1e-2:
    # update the parameter
    theta = theta - ETA * np.dot(f(X) - train_y, X)

    # calculate the difference of last time error
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

x = np.linspace(-3, 3, 100)

plt.plot(train_z, train_y, 'o')
plt.plot(x, f(to_matrix(x)))
plt.show()
