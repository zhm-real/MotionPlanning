import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as KD

PI = math.pi

WB = 3.7  # [m] wheel base: rear to front steer
LT = 8.0  # [m] rear to trailer wheel
W = 2.6  # [m] width of vehicle
LF = 4.5  # [m] distance from rear to vehicle front end of vehicle
LB = 1.0  # [m] distance from rear to vehicle back end of vehicle
LTF = 1.0  # [m] distance from rear to vehicle front end of trailer
LTB = 9.0  # [m] distance from rear to vehicle back end of trailer
MAX_STEER = 0.6  # [rad] maximum steering angle
TR = 0.5  # Tyre radius [m] for plot
TW = 1.0  # Tyre width [m] for plot

# for collision check
WBUBBLE_DIST = 3.5  # distance from rear and the center of whole bubble
WBUBBLE_R = 10.0  # whole bubble radius
B = 4.45  # distance from rear to vehicle back end
C = 11.54  # distance from rear to vehicle front end
I = 8.55  # width of vehicle
VRX = [C, C, -B, -B, C]
VRY = [-I / 2.0, I / 2.0, I / 2.0, -I / 2.0, -I / 2.0]


def calc_trailer_yaw_from_xyyaw(x, y, yaw, init_tyaw, steps):
    tyaw = [0.0 for _ in range(len(x))]
    tyaw[0] = init_tyaw

    for i in range(1, len(x)):
        tyaw[i] += tyaw[i - 1] + steps[i - 1] / LT * math.sin(yaw[i - 1] - tyaw[i - 1])

    return tyaw
