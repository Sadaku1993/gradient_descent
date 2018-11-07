import numpy as np
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import Axes3D

def gradient_vector(x, y):
    return [2.0*x/(x**2+y**2), 2.0*y/(x**2+y**2)]

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

X, Y = np.meshgrid(x, y)
Z = np.log(X**2 + Y**2)

vector = gradient_vector(1.0, 2.0)
plt.quiver(1.0, 2.0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1)

cont = plt.contour(X, Y, Z)
cont.clabel(fmt='%.1f', fontsize=14)
plt.gca().set_aspect('equal')
plt.show()

Z_plot = Z.reshape(X.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z_plot, cmap='bwr', linewidth=0)
fig.colorbar(surf)
