import unittest
import matplotlib.pyplot as plt
from assignment4 import *
from commons import *
from scipy.optimize import fsolve
import math
from numpy.random.mtrand import uniform


class TestAssignment4(unittest.TestCase):
    def test_rami(self):
        ass4 = Assignment4A()
        names = ('f', 'a', 'b', 'd', 'maxtime')
        vals = [(f1_noise, 2, 5, 1, 20),
                (f13_noise, 3, 10, 2, 15),
                (f2_noise, 0, 5, 2, 5),
                (f3_noise, -1, 5, 50, 20),
                 (f4_noise, -2, 4, 20, 10),
                 (f7_noise, 3, 16, 10, 10),
                 (f9_noise, 5, 10, 10, 20)
                 ]
        params = [dict(zip(names, val)) for val in vals]
        expected_results = [f1, f13, f2, f3, f4, f7, f9]
        i = 0
        for p in params:
            name1 = [key for (key, val) in functions.items() if val == p['f']][0].rstrip(' NOISY')
            name_noisy = [key for (key, val) in functions.items() if val == p['f']][0]
            if name1:
                print(f"########### Least Squares fitting of {name1} in range [{p['a']}, {p['b']}] ############")
            total_mse = 0
            total_time = 0
            total_fit_time = 0
            n = 1
            for _ in range(n):
                T = time.time()
                g = ass4.fit(p['f'], p['a'], p['b'], p['d'], p['maxtime'])
                fit_T = time.time() - T
                mse = 0

                for x in uniform(low=p['a'], high=p['b'], size=1000):
                    mse += (g(x) - expected_results[i](x))**2
                mse = mse / 1000
                T = time.time() - T
                total_mse += mse
                total_time += T
                total_fit_time += fit_T
                if name1:
                    plot_fit(name1, name_noisy, g, p['a'], p['b'])
            print(f'Time took to fit function: {total_fit_time / n} (Max time is: {p["maxtime"]})')
            print(f'Mean Squared Error: {total_mse / n}')
            # print(f'Time took to run test: {total_time / n}')
            print('\n')
            i += 1


def plot_fit(f_name, f_noise, f_fit, a, b):
    # name1 = [key for (key, val) in functions.items() if val == f][0]
    # name2 = [key for (key, val) in functions.items() if val == f_noise][0]
    f = functions[f_name]
    f_noise = functions[f_noise]
    plt.title(f"Fitting of {f_name}")
    x_axis = np.arange(a, b, 0.1)
    y1 = [f(x) for x in x_axis]
    y2 = [f_fit(x) for x in x_axis]
    y3 = [f_noise(x) for x in x_axis]
    plt.plot(x_axis, y1, c='r')
    plt.plot(x_axis, y2, c='g')
    plt.scatter(x_axis, y3, c='b')
    plt.show()
