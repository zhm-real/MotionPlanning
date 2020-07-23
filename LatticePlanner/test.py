import numpy as np
import math
import matplotlib.pyplot as plt

class C:
    # Parameter
    MAX_SPEED = 50.0 / 3.6
    MAX_ACCEL = 10.0
    MAX_CURVATURE = 10.0

    ROAD_WIDTH = 8.0
    ROAD_SAMPLE_STEP = 1.0

    T_STEP = 0.2
    MAX_T = 5.0
    MIN_T = 4.0

    TARGET_SPEED = 30.0 / 3.6
    SPEED_SAMPLE_STEP = 5.0 / 3.6

    # cost weights
    K_JERK = 0.1
    K_TIME = 0.1
    K_V_DIFF = 1.0
    K_OFFSET = 0.5

    # parameters for vehicle
    RF = 4.5 * 1.5  # [m] distance from rear to vehicle front end of vehicle
    RB = 1.0 * 1.5  # [m] distance from rear to vehicle back end of vehicle
    W = 3.0 * 1.5  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 3.5 * 1.5  # [m] Wheel base
    TR = 0.5 * 1.5  # [m] Tyre radius
    TW = 1 * 1.5  # [m] Tyre width
    MAX_STEER = 0.6  # [rad] maximum steering angle


def main():


if __name__ == '__main__':
    main()