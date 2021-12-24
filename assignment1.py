"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random


class Assignment1:
    def __init__(self):

        self.A = None
        self.B = None
        self.points = None
        self.interpolation = None
        self.M = np.array(
            [[-1, +3, -3, +1],
             [+3, -6, +3, 0],
             [-3, +3, 0, 0],
             [+1, 0, 0, 0]],
            dtype=np.float32
        )

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        if n == 1:
            ans = np.nan
            while np.isnan(ans):
                ans = f(np.random.uniform(a, b))
            return lambda x: f(ans)

        xs = np.linspace(a, b, n, endpoint=True)
        xs = np.where(xs == 0, -0.00005, xs)
        ys = f(xs)

        points = np.array([xs, ys]).T
        n = len(points)-1

        C = self.build_coeff_matrix(n)
        P = self.build_points_vector(points, n)

        Ax = self.solve_tridiagonal_matrix(C, [p[0] for p in P], n)
        Ay = self.solve_tridiagonal_matrix(C, [p[1] for p in P], n)
        A = np.array([Ax, Ay]).T

        B = [0] * n
        for i in range(n - 1):
            B[i] = 2 * points[i + 1] - A[i + 1]
        B[n - 1] = (A[n - 1] + points[n]) / 2

        self.A = A
        self.B = B
        self.points = points

        result = lambda x: self.get_curve_function(x)(self.normalize_x(x))[1]
        return result

    def build_coeff_matrix(self, n):
        C = 4 * np.identity(n)
        np.fill_diagonal(C[1:], 1)
        np.fill_diagonal(C[:, 1:], 1)
        C[0, 0] = 2
        C[n - 1, n - 1] = 7
        C[n - 1, n - 2] = 2
        return C

    def build_points_vector(self, points, n):
        P = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
        P[0] = points[0] + 2 * points[1]
        P[n - 1] = 8 * points[n - 1] + points[n]
        return P

    def solve_tridiagonal_matrix(self, C, P, n):
        ac = list(C.diagonal(-1))
        bc = list(C.diagonal())
        cc = list(C.diagonal(1))

        for i in range(1, n):
            mc = ac[i - 1] / bc[i - 1]
            bc[i] = bc[i] - mc * cc[i - 1]
            P[i] = P[i] - mc * P[i - 1]
        xc = bc
        xc[-1] = P[-1] / bc[-1]
        for i in range(n - 2, -1, -1):
            xc[i] = (P[i] - cc[i] * xc[i + 1]) / bc[i]
        return xc

    def get_cubic(self, p1, a, b, p2):
        return lambda t: np.power(1 - t, 3) * p1 + 3 * np.power(1 - t, 2) * t * a + 3 * (1 - t) * \
                         np.power(t, 2) * b + np.power(t, 3) * p2

    def get_curve_function(self, x):
        for i in range(len(self.points)-2, -1, -1):
            if self.points[i][0] <= x:
                return self.get_cubic(self.points[i], self.A[i], self.B[i], self.points[i+1])

    def get_curve_function2(self,x):
        keys = list(self.interpolation.keys())
        for i in range(len(keys)-1, -1, -1):
            if keys[i] <= x:
                return self.interpolation[keys[i]]

    def normalize_x(self, x):
        for i in range(len(self.points) - 2, -1, -1):
            if self.points[i][0] <= x:
                return (x - self.points[i][0])/(self.points[i+1][0] - self.points[i][0])


    def interpolate2(self, f: callable, a: float, b: float, n: int) -> callable:
        if n == 1:
            ans = np.nan
            while np.isnan(ans):
                ans = f(np.random.uniform(a, b))
            return lambda x: f(ans)

        if (n % 3)-1 != 0:
            n = 3*int(n / 3)+1

        xs = np.linspace(a, b, n, endpoint=True)
        xs = np.where(xs == 0, -0.00005, xs)
        ys = np.array([f(x) for x in xs])
        P = np.array([xs, ys]).T
        self.points = P

        # self.interpolation = {xs[i]: self.get_cubic(P[i], P[i+1], P[i+2], P[i+3]) for i in range(0, n-3, 3)}
        result = lambda x: self.inter_dot_product(x)
        return result

    def inter_dot_product(self, x):
        t = self.normalize_x(x)
        xi = None
        for i in range(len(self.points)-1, -1, -1):
            if self.points[i][0] <= x:
                xi = i
                break
        xi = int(xi / 3)*3
        p1 = self.points[xi]
        p2 = self.points[xi+1]
        p3 = self.points[xi+2]
        p4 = self.points[xi+3]
        P = np.array([p1, p2, p3, p4], dtype=np.float32)
        T = np.array([t ** 3, t ** 2, t, 1], dtype=np.float32)
        return T.dot(self.M).dot(P)[1]


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm
from mathfunctions import *


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        xxs = []
        ys = []
        yys = []
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                xxs.append(x)
                yy = ff(x)
                y = f(x)
                ys.append(y)
                yys.append(yy)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        xxs = list(sorted(xxs))
        ys = list(sorted(ys))
        yys = list(sorted(yys))
        plot_interpolation('random poly d=30', 100, xxs, ys, yys)
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


    def test_with_multyple_functions(self):
        fs = get_all_functions()
        ns = [1,10,20,50,100,200,500,1000]
        # ns = [1]
        fs = {key:val for (key,val) in fs.items() if key in ['f(x)= sin(x)/x']}
        ass1 = Assignment1()
        for f in fs:
            times = []
            errors = []
            print(f"###################### starting function {f} ################")
            for n in ns:
                print(f'n = {n}')
                T = time.time()
                ff = ass1.interpolate(fs[f], -20, 20, n)
                xs = list(sorted(np.random.uniform(-20, 20, 200)))
                err = 0
                ys = []
                yys = []
                for x in xs:
                    yy = ff(x)
                    y = fs[f](x)
                    ys.append(y)
                    yys.append(yy)
                    err += (abs(y - yy))
                errors.append(err/200)
                print(f'error = {err/200}')
                times.append(time.time() - T)
                plot_interpolation(f, n, xs, ys, yys)
            print(f'The mean error is {errors[-1]}')


from matplotlib import pyplot as plt
def plot_times(f, ns, times):
    plt.plot(ns, times)
    plt.title(f)
    plt.show()

def plot_interpolation(f,n, xs,ys,yys):
    plt.title(f + f'    n={n}')
    plt.plot(xs, ys, c='r')
    plt.plot(xs, yys, c='b')
    plt.show()
    pass

def plot_errors(f, ns, errors):
    plt.title('errors - '+f+f'   n= {ns}')
    plt.plot(ns, errors)
    plt.show()

if __name__ == "__main__":
    unittest.main()
