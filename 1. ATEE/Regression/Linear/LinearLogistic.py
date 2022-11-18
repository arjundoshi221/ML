import numpy as np


def Linear(x, y):
    np.random.seed(5)
    cost_list = []
    epoch = 100000
    lr = 0.01

    m = x.shape[0]
    nx = x.shape[1]

    add = np.ones((m, 1))
    x = np.concatenate((add, x), axis=1)
    w = np.random.rand(nx+1, 1)

    for i in range(epoch):
        h = np.dot(x, w)
        error = h-y
        dw = np.dot(x.T, error)
        w = w - (lr*dw)
        cost = (1/(2*m))*(np.dot(error.T, error))
        cost_list.append(cost[0])

    return w, cost_list


def sigmoid(z):
    a = 1/(1 + np.exp(-z))
    return a


def Linear(x, y):
    np.random.seed(5)
    cost_list = []
    epoch = 100000
    lr = 0.01

    m = x.shape[0]
    nx = x.shape[1]

    add = np.ones((m, 1))
    x = np.concatenate((add, x), axis=1)
    w = np.random.rand(nx+1, 1)

    for i in range(epoch):
        h = sigmoid(np.dot(x, w))
        error = h-y
        dw = np.dot(x.T, error)
        w = w - (lr*dw)
        logp = -(np.multiply(y, np.log(h))+np.multiply((1-y), np.log(1-h)))
        cost_list.append(logp[0])

    return w, cost_list
