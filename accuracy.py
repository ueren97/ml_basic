import numpy as np
import matplotlib.pyplot as plt

# read training data
train = np.loadtxt('sample4.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

# initialize the parameters
theta = np.random.rand(4)

# the history of accuracies
accuracies = []

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


# classification
def classify(x):
    return (f(x) >= 0.5).astype(np.int)


# repeat training
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)

    # calculate the present accuracy
    result = classify(X) == train_y
    accuracy = len(result[result == True]) / len(result)
    accuracies.append(accuracy)


# plot accuracies
x = np.arange(len(accuracies))

plt.plot(x, accuracies)
plt.show()
