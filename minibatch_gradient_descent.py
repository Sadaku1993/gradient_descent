#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

m = 100
n_epochs = 50 # エポック数
batch_size = 20 # バッチサイズ

t0, t1 = 200, 1000
def learning_schedule(t):
    return t0/float(t+t1)

X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
X_b = np.c_[np.ones((m, 1)), X]  # add x0 = 1 to each instance

theta = np.random.randn(2, 1)

plt.plot(X, y, "ob")

t = 0
for epoch in range(n_epochs):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, batch_size):
        t += 1
        xi = X_b_shuffled[i:i+batch_size]
        yi = y_shuffled[i:i+batch_size]
        gradients = 2.0/batch_size*xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta*gradients
        
        if(i<20):
            X_new = np.array([[0], [2]])
            X_new_b = np.c_[np.ones((2, 1)), X_new]
            y_predict = X_new_b.dot(theta)
            plt.plot(X_new, y_predict, "b-")

plt.xlabel("x")
plt.ylabel("y")
plt.axis([0, 2, 0, 15])
plt.savefig("minibatch_gradient_descent")
plt.show()
