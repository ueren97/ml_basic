import numpy as np
import matplotlib.pyplot as plt

# truth function


def g(x):
    return 0.1 * (x ** 3 + x ** 2 + x)

# ----- not reqularized parameters -----


# prepare for training date noized date is added to truth funciton
train_x = np.linspace(-2, 2, 8)
train_y = g(train_x) + np.random.randn(train_x.size) * 0.05

x = np.linspace(-2, 2, 100)

# standardization
mu = train_x.mean()
sigma = train_x.std()


def standardize(x):
    return (x - mu) / sigma


train_z = standardize(train_x)

# create matrix of training data


def to_matrix(x):
    return np.vstack([
        np.ones(x.size),
        x,
        x ** 2,
        x ** 3,
        x ** 4,
        x ** 5,
        x ** 6,
        x ** 7,
        x ** 8,
        x ** 9,
        x ** 10,
    ]).T


X = to_matrix(train_z)

# initialize the parameters
theta = np.random.randn(X.shape[1])


# prediction function
def f(x):
    return np.dot(x, theta)

# objective funciton


def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)


# estimated time of arrival
ETA = 1e-4

# error
diff = 1

# repeat training
error = E(X, train_y)
while diff > 1e-6:
    theta = theta - ETA * np.dot(f(X) - train_y, X)
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

# standarize x
z = standardize(x)

# ----------

# --- regularized parameters ---

# initialize the parameter again after saving not regularized parameter
theta1 = theta
theta = np.random.randn(X.shape[1])

# regularization constant
LAMBDA = 1

# error
diff = 1

# repeat training (with regularization)
error = E(X, train_y)
while diff > 1e-6:
    # regularized term. Bias term will be 0 (not regularized)
    reg_term = LAMBDA * np.hstack([0, theta[1:]])

    # update the parameters by regularizing
    theta = theta - ETA * (np.dot(f(X) - train_y, X) + reg_term)
    current_error = E(X, train_y)
    diff = error - current_error
    error = current_error

# ---------


# compare

# save regularized parameters
theta2 = theta
plt.plot(train_z, train_y, 'o')

# plot not regularized result
theta = theta1
plt.plot(z, f(to_matrix(z)), linestyle='dashed')

# plot regularized result
theta = theta2
plt.plot(z, f(to_matrix(z)))


plt.show()
