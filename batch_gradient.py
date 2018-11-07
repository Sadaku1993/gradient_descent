#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

eta = 0.1 # 学習率
n_iterations = 100 # エポック数
m = 100 # データ数

X = 2 * np.random.rand(m, 1)
y = 4 + 3 * X + np.random.randn(m, 1)
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance

theta = np.random.randn(2, 1)

plt.plot(X, y, "ob")

for iterations in range(n_iterations):
    gradients = 2/float(m) * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict = X_new_b.dot(theta)
    
    if(iterations<10):
        plt.plot(X_new, y_predict, "b-")
    elif(iterations==n_iterations-1):
        plt.plot(X_new, y_predict, "r-")

plt.xlabel("x")
plt.ylabel("y")
plt.title(r"$\eta={}$".format(eta), fontsize=16)
plt.axis([0, 2, 0, 15])
plt.savefig("batch_gradient")
plt.show()
