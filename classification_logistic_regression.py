import numpy as np
import matplotlib.pyplot as plt

# read training data
train = np.loadtxt('sample3.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

# initialize parameter
theta = np.random.rand(3)

# standardization
mu = train_x.mean(axis=0)
sigma = train_x.std(axis=0)


def standardize(x):
    return (x - mu) / sigma


train_z = standardize(train_x)


# add x0

def to_matrix(x):
    x0 = np.ones([x.shape[0], 1])
    return np.hstack([x0, x])


X = to_matrix(train_z)


# sigmoid function
def f(x):
    return 1 / (1 + np.exp(-np.dot(x, theta)))


# classification function
def classify(x):
    return (f(x) >= 0.5).astype(np.int)


# estimated time of arrival
ETA = 1e-3

# repeat
epoch = 5000

# the update number of times
count = 0

# repeat training
for _ in range(epoch):
    theta = theta - ETA * np.dot(f(X) - train_y, X)

    # output log
    count += 1
    print('{} times: theta = {}'.format(count, theta))

# confirm by plotting
x0 = np.linspace(-2, 2, 100)
plt.plot(train_z[train_y == 1, 0], train_z[train_y == 1, 1], 'o')
plt.plot(train_z[train_y == 0, 0], train_z[train_y == 0, 1], 'x')
plt.plot(x0, -(theta[0] + theta[1] * x0) / theta[2], linestyle='dashed')
plt.show()
