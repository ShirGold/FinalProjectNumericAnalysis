"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""
import matplotlib.pyplot as plt
import numpy
import numpy as np
import time
import random
from assignment1 import *


class Assignment4A:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        pass

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        points = self.get_samples(f, a, b, maxtime)
        result = self.linear_least_squares(points, d)
        # result = self.alternative_least_squared(points, d)
        return result

    def linear_least_squares(self, samples, d):
        A = numpy.vander(samples[0], d+1, increasing=False)
        y = samples[1]
        c = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose().dot(y))
        poly = np.polynomial.Polynomial(np.flip(c))
        return poly

    def alternative_least_squared(self, samples, d):
        A = numpy.vander(samples[0], d+1, increasing=False)
        ex = np.exp(samples[0])
        ex2 = 1/np.exp(samples[0])
        sin = np.sin(samples[0])
        cos = np.cos(samples[0])
        sin2 = np.sin(np.power(samples[0], 2))
        cos2 = np.cos(np.power(samples[0], 2))
        y = samples[1]
        A = np.c_[A, ex, ex2, sin, sin2, cos, cos2]
        c = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose().dot(y))
        poly = np.polynomial.Polynomial(np.flip(c[:-6]))
        return lambda x: poly(x) + c[-6]*np.exp(x) + c[-5]*(1/np.exp(x)) + c[-4]*np.sin(x) + c[-3]*np.sin(x**2) + c[-2]*np.cos(x) + c[-1]*np.cos(x**2)

    def get_samples(self, f, a, b, maxtime):
        T = time.time()
        y = f(a)
        delta = time.time()-T
        if delta > maxtime/2:
            return a, y
        if delta == 0:
            n = 100000
            mult = 900
        else:
            n = int(maxtime*2/delta)
            mult = max(int(1/(2*delta)), 1)
        if n <= 1:
            return a, y
        xs = np.linspace(a, b, n, endpoint=True)
        mat = []
        for _ in range(0, mult):
            mat.append(f(xs))
        mat = np.array(mat)
        ys = mat.mean(axis=0)

        return np.array([xs, ys])



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(4)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEqual(f(x), nf(x))
            mse += (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)

    def test_err3(self):
        f = poly(2, 1, 1, 1, 1, 6, 1, 3, 3, 9, 13, 5, 3)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=-2, b=2, d=10, maxtime=5)
        T = time.time() - T
        print(T)
        mse=0
        xs = np.linspace(-2, 2, 10000)
        y = []
        yy = []
        for x in xs:
            y.append(f(x))
            yy.append(ff(x))
            self.assertNotEqual(f(x), nf(x))
            mse += (f(x)-ff(x))**2
        plt.plot(xs, y, c='g')
        plt.scatter(xs, yy, c='r')
        plt.show()
        mse = mse/10000
        print(mse)

    def test_err2(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEqual(f(x), nf(x))
            mse += (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)

    def test_multyple_values_for_function(self):
        pass


if __name__ == "__main__":
    unittest.main()
