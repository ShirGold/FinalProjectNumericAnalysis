"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""
import math

import numpy as np
import time
import random
from assignment2 import Assignment2


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        # replace this line with your solution
        c = 1
        if a > b:
            c = -1
            temp = b
            b = a
            a = temp
        result = self.composite_simpsons_rule(f, a, b, n)

        return result*c

    def composite_simpsons_rule(self, f: callable, a, b, n):
        if n % 2 == 1:
            n -= 1
        h = (b-a)/n
        xs = np.linspace(a, b, n+1, endpoint=True)
        ys = f(xs)
        res = np.float32(h/3*np.sum(ys[0:-1:2] + 4*ys[1::2] + ys[2::2]))

        return res

    def alternative_integral(self, f, a, b, n):
        if n % 4 != 0:
            n -= n % 4
        h = (b - a) / n
        xs = np.linspace(a, b, n+1, endpoint=True)
        ys = f(xs)
        res = np.float32(h / 90 * np.sum(7*ys[0:-4:4]+32*ys[1::4]+12*ys[2::4]+32*ys[3::4]+7*ys[4::4]))

        return res



    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        # replace this line with your solution
        ass2 = Assignment2()
        intersections = ass2.intersections(f1, f2, 1, 100)
        g = lambda x: abs(f1(x) - f2(x))
        if len(intersections) < 2:
            return np.nan
        a = intersections[0]
        b = intersections[-1]
        result = self.integrate(g, a, b, 1000)

        return result



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm
from functionUtils import *


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEqual(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_integral(self):
        ass3 = Assignment3()
        f = RESTRICT_INVOCATIONS(20)(np.polynomial.Polynomial([3, -1, 1, 1]))
        r = ass3.integrate(f, 1, 10, 20)
        true_result = 11241/4
        self.assertGreaterEqual(0.01, abs((r - true_result) / true_result))

    def test_area_between_polys(self):
        ass3 = Assignment3()
        f1 = np.polynomial.Polynomial([1, 0, -2, 1])
        f2 = np.polynomial.Polynomial([0, 1])
        r = ass3.areabetween(f1, f2)
        true_result = 2.76555
        self.assertGreaterEqual(0.001, abs((r - true_result)/true_result))

    def test_area_between_not_polys(self):
        ass3 = Assignment3()
        f1 = lambda x: np.sin(3*x)*5
        f2 = lambda x: (x-20)/3
        r = ass3.areabetween(f1, f2)
        true_result = 104.088
        self.assertGreaterEqual(0.001, abs((r - true_result)/true_result))

    def test_stack_funcs(self):
        ass = Assignment3()
        f1 = lambda x: 5*np.exp(-1*(x**2))
        a1 = 0
        b1 = 4
        f2 = lambda x: np.log(6*np.log(x))
        a2 = 2
        b2 = 5
        f3 = lambda x: (np.exp(-1*x))/x
        f4 = lambda x: (x**(math.e - 1))*np.exp(-1*x)
        print(f'f1 integral: {ass.integrate(f1, a1, b1, 4)}')
        print(f'f2 integral: {ass.integrate(f2, a2, b2, 4)}')
        print(f'f3 integral: {ass.integrate(f3, a2, b2, 4)}')
        print(f'f4 integral: {ass.integrate(f4, a2, b2, 4)}')


if __name__ == "__main__":
    unittest.main()
