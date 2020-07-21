import math
import numpy as np
import matplotlib.pyplot as plt

from CurvesGenerator import draw


class QuinticPolynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, time):
        # calc coefficient of quintic polynomial
        # See jupyter notebook document for derivation of this equation.
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[time ** 3, time ** 4, time ** 5],
                      [3 * time ** 2, 4 * time ** 3, 5 * time ** 4],
                      [6 * time, 12 * time ** 2, 20 * time ** 3]])
        b = np.array([xe - self.a0 - self.a1 * time - self.a2 * time ** 2,
                      vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt


# def simulation():
#     sx, sy, syaw, sv, sa = 10.0, 10.0, np.deg2rad(10.0), 1.0, 0.1
#     gx, gy, gyaw, gv, ga = 30.0, -10.0, np.deg2rad(20.0), 1.0, 0.1
#
#     MAX_ACCEL = 1.0  # max accel [m/ss]
#     MAX_JERK = 0.5  # max jerk [m/sss]
#     dt = 0.1  # T tick [s]
#
#     MIN_T = 5
#     MAX_T = 100
#     T_STEP = 5
#
#     sv_x = sv * math.cos(syaw)
#     sv_y = sv * math.sin(syaw)
#     gv_x = gv * math.cos(gyaw)
#     gv_y = gv * math.sin(gyaw)
#
#     sa_x = sa * math.cos(syaw)
#     sa_y = sa * math.sin(syaw)
#     ga_x = ga * math.cos(gyaw)
#     ga_y = ga * math.sin(gyaw)
#
#     trec, xrec, yrec, yawrec, vrec, arec, jerkrec = [], [], [], [], [], [], []
#
#     for T in np.arange(MIN_T, MAX_T, T_STEP):
#         xqp = QuinticPolynomial(sx, sv_x, sa_x, gx, gv_x, ga_x, T)
#         yqp = QuinticPolynomial(sy, sv_y, sa_y, gy, gv_y, ga_y, T)
#
#         trec, xrec, yrec, yawrec, vrec, arec, jerkrec = [], [], [], [], [], [], []
#
#         for t in np.arange(0.0, T + dt, dt):
#             trec.append(t)
#             xrec.append(xqp.calc_xt(t))
#             yrec.append(yqp.calc_xt(t))
#
#             vx = xqp.calc_dxt(t)
#             vy = yqp.calc_dxt(t)
#             vrec.append(np.hypot(vx, vy))
#             yawrec.append(math.atan2(vy, vx))
#
#             ax = xqp.calc_ddxt(t)
#             ay = yqp.calc_ddxt(t)
#             a = np.hypot(ax, ay)
#
#             if len(vrec) >= 2 and vrec[-1] - vrec[-2] < 0.0:
#                 a *= -1
#             arec.append(a)
#
#             jx = xqp.calc_dddxt(t)
#             jy = yqp.calc_dddxt(t)
#             j = np.hypot(jx, jy)
#
#             if len(arec) >= 2 and arec[-1] - arec[-2] < 0.0:
#                 j *= -1
#             jerkrec.append(j)
#
#         if max([abs(i) for i in arec]) <= MAX_ACCEL and \
#                 max([abs(i) for i in jerkrec]) <= MAX_JERK:
#             print("find path!!")
#             break
#
#     print("t_len: ", trec[-1], "s")
#     print("max_v: ", max(vrec), "m/s")
#     print("max_a: ", max(np.abs(arec)), "m/s2")
#     print("max_jerk: ", max(np.abs(jerkrec)), "m/s3")
#
#     for i in range(len(trec)):
#         plt.cla()
#         plt.gcf().canvas.mpl_connect('key_release_event',
#                                      lambda event: [exit(0) if event.key == 'escape' else None])
#         plt.axis("equal")
#         plt.plot(xrec, yrec, linewidth=2, color='gray')
#         draw.Car(sx, sy, syaw, 1.5, 3)
#         draw.Car(gx, gy, gyaw, 1.5, 3)
#         draw.Car(xrec[i], yrec[i], yawrec[i], 1.5, 3)
#         plt.title("Quintic Polynomial Curves")
#         plt.pause(0.001)
#
#     plt.show()


# if __name__ == '__main__':

