import numpy as np
import math
import matplotlib.pyplot as plt

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


def check_collision(x, y, yaw, kdtree, ox, oy, wbd, wbr, vrx, vry):
    for ix, iy, iyaw in zip(x, y, yaw):
        cx = ix + wbd * math.cos(iyaw)
        cy = iy + wbd * math.sin(iyaw)

        ids = inrange(kdtree, [cx, cy], wbr)

        if len(ids) == 0:
            continue

        obsx, obsy = [], []
        for i in ids:
            obsx.append(ox[i])
            obsy.append(oy[i])

        if not rect_check(ix, iy, iyaw, obsx, obsy, vrx, vry):
            return False

    return True


def inrange(kdtree, c, wbr):
    ind_list = []

    for i in range(len(kdtree)):
        if math.hypot(kdtree[i][0] - c[0], kdtree[i][1] - c[1]) < wbr:
            ind_list.append(i)

    return ind_list


def rect_check(ix, iy, iyaw, ox, oy, vrx, vry):
    c = math.cos(-iyaw)
    s = math.sin(-iyaw)

    for (iox, ioy) in zip(ox, oy):
        tx = iox - ix
        ty = ioy - iy
        lx = (c * tx - s * ty)
        ly = (s * tx + c * ty)

        sumangle = 0.0

        for i in range(len(vrx) - 1):
            x1 = vrx[i] - lx
            y1 = vry[i] - ly
            x2 = vrx[i + 1] - lx
            y2 = vry[i + 1] - ly
            d1 = math.hypot(x1, y1)
            d2 = math.hypot(x2, y2)
            theta1 = math.atan2(y1, x1)
            tty = (-math.sin(theta1) * x2 + math.cos(theta1) * y2)
            tmp = (x1 * x2 + y1 * y2) / (d1 * d2)

            if tmp >= 1.0:
                tmp = 1.0
            elif tmp <= 0.0:
                tmp = 0.0

            if tty >= 0.0:
                sumangle += math.acos(tmp)
            else:
                sumangle -= math.acos(tmp)

        if sumangle >= PI:
            return False

    return True


def calc_trailer_yaw_from_xyyaw(x, y, yaw, init_tyaw, steps):
    tyaw = [0.0 for _ in range(len(x))]
    tyaw[0] = init_tyaw

    for i in range(1, len(x)):
        tyaw[i] += tyaw[i - 1] + steps[i - 1] / LT * math.sin(yaw[i - 1] - tyaw[i - 1])

    return tyaw


def trailer_motion_model(x, y, yaw0, yaw1, D, d, L, delta):
    x += D * math.cos(yaw0)
    y += D * math.sin(yaw0)
    yaw0 += D / L * math.tan(delta)
    yaw1 += D / d * math.sin(yaw0 - yaw1)

    return x, y, yaw0, yaw1


def check_trailer_collision(ox, oy, x, y, yaw0, yaw1, kdtree=None):
    if not kdtree:
        kdtree = []
        for i in range(len(ox)):
            kdtree.append((ox[i], oy[i]))

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


def plot_trailer(x, y, yaw, yaw1, steer):
    truckcolor = "-k"

    LENGTH = LB + LF
    LENGTHt = LTB + LTF

    truckOutLine = np.array([[-LB, (LENGTH - LB), (LENGTH - LB), (-LB), (-LB)],
                             [W / 2, W / 2, -W / 2, -W / 2, W / 2]])

    trailerOutLine = np.array([[-LTB, (LENGTHt - LTB), (LENGTHt - LTB), (-LTB), (-LTB)],
                               [W / 2, W / 2, -W / 2, -W / 2, W / 2]])

    rr_wheel = np.array([[TR, -TR, -TR, TR, TR],
                         [-W / 12.0 + TW, -W / 12.0 + TW, W / 12.0 + TW, W / 12.0 + TW, -W / 12.0 + TW]])

    rl_wheel = np.array([[TR, -TR, -TR, TR, TR],
                         [-W / 12.0 - TW, -W / 12.0 - TW, W / 12.0 - TW, W / 12.0 - TW, -W / 12.0 - TW]])

    fr_wheel = np.array([[TR, -TR, -TR, TR, TR],
                         [-W / 12.0 + TW, -W / 12.0 + TW, W / 12.0 + TW, W / 12.0 + TW, -W / 12.0 + TW]])

    fl_wheel = np.array([[TR, -TR, -TR, TR, TR],
                         [-W / 12.0 - TW, -W / 12.0 - TW, W / 12.0 - TW, W / 12.0 - TW, -W / 12.0 - TW]])

    tr_wheel = np.array([[TR, -TR, -TR, TR, TR],
                         [-W / 12.0 + TW, -W / 12.0 + TW, W / 12.0 + TW, W / 12.0 + TW, -W / 12.0 + TW]])

    tl_wheel = np.array([[TR, -TR, -TR, TR, TR],
                         [-W / 12.0 - TW, -W / 12.0 - TW, W / 12.0 - TW, W / 12.0 - TW, -W / 12.0 - TW]])

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])

    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    Rot3 = np.array([[math.cos(yaw1), math.sin(yaw1)],
                     [-math.sin(yaw1), math.cos(yaw1)]])

    fr_wheel = (np.dot(fr_wheel.T, Rot2)).T
    fl_wheel = (np.dot(fl_wheel.T, Rot2)).T

    fr_wheel[0, :] = fr_wheel[0, :] + WB
    fl_wheel[0, :] = fl_wheel[0, :] + WB

    fr_wheel = (np.dot(fr_wheel.T, Rot1)).T
    fl_wheel = (np.dot(fl_wheel.T, Rot1)).T

    tr_wheel[0, :] = tr_wheel[0, :] - LT
    tl_wheel[0, :] = tl_wheel[0, :] - LT
    tr_wheel = (np.dot(tr_wheel.T, Rot3)).T
    tl_wheel = (np.dot(tl_wheel.T, Rot3)).T

    truckOutLine = (np.dot(truckOutLine.T, Rot1)).T
    trailerOutLine = (np.dot(trailerOutLine.T, Rot3)).T
    rr_wheel = (np.dot(rr_wheel.T, Rot1)).T
    rl_wheel = (np.dot(rl_wheel.T, Rot1)).T

    truckOutLine[0, :] += x
    truckOutLine[1, :] += y
    trailerOutLine[0, :] += x
    trailerOutLine[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    tr_wheel[0, :] += x
    tr_wheel[1, :] += y
    tl_wheel[0, :] += x
    tl_wheel[1, :] += y

    plt.plot(truckOutLine[0, :], truckOutLine[1, :], truckcolor)
    plt.plot(trailerOutLine[0, :], trailerOutLine[1, :], truckcolor)
    plt.plot(fr_wheel[0, :], fr_wheel[1, :], truckcolor)
    plt.plot(rr_wheel[0, :], rr_wheel[1, :], truckcolor)
    plt.plot(fl_wheel[0, :], fl_wheel[1, :], truckcolor)
    plt.plot(rl_wheel[0, :], rl_wheel[1, :], truckcolor)

    plt.plot(tr_wheel[0, :], tr_wheel[1, :], truckcolor)
    plt.plot(tl_wheel[0, :], tl_wheel[1, :], truckcolor)
    plt.plot(x, y, "*")


def main():
    x = 0.0
    y = 0.0
    yaw0 = np.deg2rad(10.0)
    yaw1 = np.deg2rad(-10.0)

    plot_trailer(x, y, yaw0, yaw1, 0.0)

    DF = (LF + LB) / 2.0 - LB
    DFR = (LF + LB) / 2.0 + 0.3

    DT = (LTF + LTB) / 2.0 - LTB
    DTR = (LTF + LTB) / 2.0 + 0.3

    plt.axis("equal")

    plt.show()


if __name__ == '__main__':
    main()