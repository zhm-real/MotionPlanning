"""
original author: Atsushi Sakai (@Atsushi_twi)
modified: huiming zhou
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from CurvesGenerator import draw


class QuinticPolynomial:
    def __init__(self, x0, dx0, ddx0, x1, dx1, ddx1, T):
        A = np.array([[T ** 3, T ** 4, T ** 5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([x1 - (x0 + dx0 * T + 0.5 * ddx0 * T ** 2),
                      dx1 - (dx0 + ddx0 * T),
                      ddx1 - ddx0])
        X = np.linalg.solve(A, b)

        self.a0 = x0
        self.a1 = dx0
        self.a2 = ddx0 / 2.0
        self.a3 = X[0]
        self.a4 = X[1]
        self.a5 = X[2]
        self.T = T

    def calc_t(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5
        return xt

    def calc_dt(self, t):
        dxt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return dxt

    def calc_ddt(self, t):
        ddxt = 2 * self.a2 + 6 * self.a3 * t + \
              12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return ddxt

    def calc_dddt(self, t):
        dddxt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return dddxt


def simulation():
    sx, sy, syaw, sv, sa = 10.0, 10.0, np.deg2rad(10.0), 1.0, 0.1
    gx, gy, gyaw, gv, ga = 30.0, -10.0, np.deg2rad(20.0), 1.0, 0.1

    MAX_ACCEL = 1.0  # max accel [m/ss]
    MAX_JERK = 0.5  # max jerk [m/sss]
    dt = 0.1  # time tick [s]

    MIN_T = 5
    MAX_T = 100
    T_STEP = 5

    sv_x = sv * math.cos(syaw)
    sv_y = sv * math.sin(syaw)
    gv_x = gv * math.cos(gyaw)
    gv_y = gv * math.sin(gyaw)

    sa_x = sa * math.cos(syaw)
    sa_y = sa * math.sin(syaw)
    ga_x = ga * math.cos(gyaw)
    ga_y = ga * math.sin(gyaw)

    trec, xrec, yrec, yawrec, vrec, arec, jerkrec = [], [], [], [], [], [], []

    for T in np.arange(MIN_T, MAX_T, T_STEP):
        xqp = QuinticPolynomial(sx, sv_x, sa_x, gx, gv_x, ga_x, T)
        yqp = QuinticPolynomial(sy, sv_y, sa_y, gy, gv_y, ga_y, T)

        trec, xrec, yrec, yawrec, vrec, arec, jerkrec = [], [], [], [], [], [], []

        for t in np.arange(0.0, T + dt, dt):
            trec.append(t)
            xrec.append(xqp.calc_t(t))
            yrec.append(yqp.calc_t(t))

            vx = xqp.calc_dt(t)
            vy = yqp.calc_dt(t)
            vrec.append(np.hypot(vx, vy))
            yawrec.append(math.atan2(vy, vx))

            ax = xqp.calc_ddt(t)
            ay = yqp.calc_ddt(t)
            a = np.hypot(ax, ay)

            if len(vrec) >= 2 and vrec[-1] - vrec[-2] < 0.0:
                a *= -1
            arec.append(a)

            jx = xqp.calc_dddt(t)
            jy = yqp.calc_dddt(t)
            j = np.hypot(jx, jy)

            if len(arec) >= 2 and arec[-1] - arec[-2] < 0.0:
                j *= -1
            jerkrec.append(j)

        if max([abs(i) for i in arec]) <= MAX_ACCEL and \
                max([abs(i) for i in jerkrec]) <= MAX_JERK:
            print("find path!!")
            break

    print("t_len: ", trec[-1], "s")
    print("max_v: ", max(vrec), "m/s")
    print("max_a: ", max(np.abs(arec)), "m/s2")
    print("max_jerk: ", max(np.abs(jerkrec)), "m/s3")

    for i in range(len(trec)):
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.axis("equal")
        plt.plot(xrec, yrec, linewidth=2, color='gray')
        draw.Car(sx, sy, syaw, 1.5, 3)
        draw.Car(gx, gy, gyaw, 1.5, 3)
        draw.Car(xrec[i], yrec[i], yawrec[i], 1.5, 3)
        plt.title("Quintic Polynomial Curves")
        plt.pause(0.001)

    plt.show()


if __name__ == '__main__':
    simulation()
