import numpy as np
import math

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


def check_trailer_collision(ox, oy, x, y, yaw0, yaw1, kdtree=None):
    if not kdtree:
        kdtree = set()
        for i in range(len(ox)):
            kdtree.add((ox[i], oy[i]))

    vrxt = [LTF, LTF, -LTB, -LTB, LTF]
    vryt = [-W / 2.0, W / 2.0, W / 2.0, -W / 2.0, -W / 2.0]

    # bubble parameter
    DT = (LTF + LTB) / 2.0 - LTB
    DTR = (LTF + LTB) / 2.0 + 0.3

    # check trailer
    if not check_collision(x, y, yaw1, kdtree, ox, oy, DT, DTR, vrxt, vryt):
        return False

    vrxf = [LF, LF, -LB, -LB, LF]
    vryf = [-W / 2.0, W / 2.0, W / 2.0, -W / 2.0, -W / 2.0]

    # bubble parameter
    DF = (LF + LB) / 2.0 - LB
    DFR = (LF + LB) / 2.0 + 0.3

    # check front trailer
    if not check_collision(x, y, yaw0, kdtree, ox, oy, DF, DFR, vrxf, vryf):
        return False

    return True  # OK


def check_collision(x, y, yaw, kdtree, ox, oy, wbd, wbr, vrx, vry):
    for ix, iy, iyaw in zip(x, y, yaw):
        cx = ix + wbd*math.cos(iyaw)
        cy = iy + wbd*math.sin(iyaw)

        