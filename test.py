import numpy as np
import math
import matplotlib.pyplot as plt

car = np.array([[-2, -2, 2, 2, -2],
                [1, -1, -1, 1, 1]])
plt.plot(car[0, :], car[1, :], '-k')
theta = np.deg2rad(30)
R = np.array([[math.cos(theta), -math.sin(theta)],
              [math.sin(theta), math.cos(theta)]])

car = np.dot(R, car)
plt.plot(car[0, :], car[1, :], '-k')
plt.axis("equal")
plt.show()
