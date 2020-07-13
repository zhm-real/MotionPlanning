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
    plt.axis("equal")
    plt.show()


if __name__ == '__main__':
    main()