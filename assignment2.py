"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable
from matplotlib import pyplot as plt

class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        if isinstance(f1, np.poly1d) or isinstance(f1, np.polynomial.Polynomial):
            if isinstance(f1, np.poly1d) or isinstance(f1, np.polynomial.Polynomial):
                g = f1 - f2
            else:
                g = lambda x: f1(x) - f2(x)
        else:
            g = lambda x: f1(x) - f2(x)
        xs = self.get_starting_points(g, a, b, maxerr)
        X = []
        for x in xs:
            root = self.get_newthon_rapson_root(g, x, maxerr)
            if root is not None:
                X.append(root)
        if X[-1] > b:
            X.pop(-1)

        X = self.filter_close_roots(X, maxerr)

        # xxs = np.linspace(X[0], X[-1], 2000)
        # y1s = np.array([f1(x) for x in xxs])
        # y2s = np.array([f2(x) for x in xxs])
        # yys = np.array([f1(x) for x in X])
        # plt.plot(xxs, y1s)
        # plt.plot(xxs, y2s, c='black')
        # plt.scatter(X, yys, c='r')
        # plt.show()

        return X

    def get_derivative(self, f: callable, x, dx=1e-6):
        return (f(x+dx)-f(x-dx))/(2*dx)

    def get_newthon_rapson_root(self, f: callable, x, maxerror, maxit=100):
        fx = f(x)
        for _ in range(maxit):
            if abs(fx) < maxerror:
                return x
            der = self.get_derivative(f, x)
            if abs(der) < maxerror:
                break
            x = x - fx/der
            fx = f(x)
        return None

    def get_starting_points(self, f: callable, a, b, maxerror):
        if isinstance(f, np.poly1d):
            n = f.order*100
        elif isinstance(f, np.polynomial.Polynomial):
            n = f.degree()*100
        else:
            n = int((abs(b-a)/maxerror)/2)
        xs = np.linspace(a, b, n, endpoint=True)
        ys = f(xs)
        ders = np.array([self.get_derivative(f, x) for x in xs])
        moment = 0
        starting_points = []
        for i in range(len(ders)):
            if ders[i] > 0 and moment <= 0 and ys[i] < 0:
                moment = 1
                starting_points.append(xs[i])
            elif ders[i] < 0 and moment >= 0 and ys[i] > 0:
                moment = -1
                starting_points.append(xs[i])
            elif ders[i] > 0 and moment == 1 and ys[i] < 0:
                starting_points[-1] = xs[i]
            elif ders[i] < 0 and moment == -1 and ys[i] > 0:
                starting_points[-1] = xs[i]
        if len(starting_points) < 2:
            return [a, b]
        return starting_points

    def filter_close_roots(self, roots, err):
        i = 0
        end = len(roots)-1
        while i < end:
            if abs(roots[i] - roots[i+1]) < err:
                roots.pop(i)
                i -= 1
                end -= 1
            i += 1
        return roots



##########################################################################

import unittest
from sampleFunctions import *
from tqdm import tqdm
import mathfunctions


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_multiple_intersections(self):
        ass2 = Assignment2()

        f1 = mathfunctions.function5
        f2 = lambda x: 0
        X = ass2.intersections(f1, f2, -30, 30, maxerr=0.001)
        print(X)
        print(len(X))
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_multiple_intersections2(self):
        ass2 = Assignment2()

        f1 = mathfunctions.function3
        f2 = lambda x: 0
        X = ass2.intersections(f1, f2, -5, 5, maxerr=0.001)
        print(X)
        print(len(X))
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_multiple_intersections3(self):
        ass2 = Assignment2()

        f1 = mathfunctions.function3
        f2 = lambda x: np.sin(3*x)
        X = ass2.intersections(f1, f2, -20, -10, maxerr=0.001)
        print(X)
        print(len(X))
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_multiple_intersections4(self):
        ass2 = Assignment2()

        f1 = lambda x: np.cos(x)
        f2 = lambda x: np.sin(x)
        X = ass2.intersections(f1, f2, -100, 100, maxerr=0.001)
        print(X)
        print(len(X))
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
