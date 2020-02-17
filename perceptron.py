import numpy as np
import matplotlib.pyplot as plt

# read training data
train = np.loadtxt('sample2.csv', delimiter=',', skiprows=1)
train_x = train[:, 0:2]
train_y = train[:, 2]

# initialize the weight(w)
w = np.random.rand(2)

# identification function


def f(x):
    if np.dot(w, x) >= 0:
        return 1
    else:
        return -1


# repeat
epoch = 10

# the update number of times
count = 10

# train the weight
for _ in range(epoch):
    for x, y in zip(train_x, train_y):
        if f(x) != y:
            w = w + y * x

            # outoput log
            count += 1
            print('{}times: w = {}'.format(count, w))


# plotting
x1 = np.arange(0, 500)

plt.plot(train_x[train_y == 1, 0], train_x[train_y == 1, 1], 'o')
plt.plot(train_x[train_y == -1, 0], train_x[train_y == -1, 1], 'x')
plt.plot(x1, -w[0] / w[1] * x1, linestyle='dashed')
plt.show()
