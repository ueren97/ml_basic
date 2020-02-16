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


# initialize parameters
theta0 = np.random.rand()
theta1 = np.random.rand()


# prediction function
def f(x):
    return theta0 + theta1 * x


# objective function
def E(x, y):
    return 0.5 * np.sum((y - f(x)) ** 2)


# estimated time of arrival(ETA)
ETA = 1e-3


# difference of error
diff = 1


# update number of times
count = 0


# repeat update till the difference of error < 0.01
error = E(train_z, train_y)
while diff > 1e-2:
    # save an update result into a temporary valiable
    tmp_theta0 = theta0 - ETA * np.sum((f(train_z) - train_y))
    tmp_theta1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)

    # update the parameters
    theta0 = tmp_theta0
    theta1 = tmp_theta1

    # calculate the difference of last time error
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error

    # output log
    count += 1
    log = '{} times: theta0 = {:.3f}, theta1 = {:.3f}, difference = {:.4f}'
    print(log.format(count, theta0, theta1, diff))


# comfirm by plotting data
x = np.linspace(-3, 3, 100)
plt.plot(train_z, train_y, 'o')
plt.plot(x, f(x))
plt.show()
