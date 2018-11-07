#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_epochs = 50 # エポック数
m = 100 # データ数

t0, t1 = 5, 50
def learning_schedule(t):
    return t0/float(t+t1)

X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
X_b = np.c_[np.ones((m, 1)), X]  # add x0 = 1 to each instance

theta = np.random.randn(2, 1)

plt.plot(X, y, "ob")

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2.0 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch*m + i)
        theta = theta - eta * gradients
        
        if(epoch==0 and i<20):
            X_new = np.array([[0], [2]])
            X_new_b = np.c_[np.ones((2, 1)), X_new]
            y_predict = X_new_b.dot(theta)
            plt.plot(X_new, y_predict, "b-")

plt.xlabel("x")
plt.ylabel("y")
plt.axis([0, 2, 0, 15])
plt.savefig("stochastic_gradient_descent")
plt.show()
