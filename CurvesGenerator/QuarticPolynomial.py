import numpy as np


class QuarticPolynomial:

    def __init__(self, x0, dx0, ddx0, dx1, ddx1, T):
        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([dx1 - dx0 - ddx0 * T,
                      ddx1 - ddx0])
        X = np.linalg.solve(A, b)

        self.a0 = x0
        self.a1 = dx0
        self.a2 = ddx0 / 2.0
        self.a3 = X[0]
        self.a4 = X[1]

    def calc_xt(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_dxt(self, t):
        dxt = self.a1 + 2 * self.a2 * t + \
              3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return dxt

    def calc_ddxt(self, t):
        ddxt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return ddxt

    def calc_dddxt(self, t):
        dddxt = 6 * self.a3 + 24 * self.a4 * t

        return dddxt
