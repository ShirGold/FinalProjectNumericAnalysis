"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import time
import random

import pandas as pd

from functionUtils import AbstractShape
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import ConvexHull, Delaunay, delaunay_plot_2d, distance
from commons import *
from concurrent.futures import ThreadPoolExecutor
from assignment1 import Assignment1
import polylidar
import seaborn as sns

class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, vertices, hull):
        self.vertices = vertices
        self.ar = np.float32(hull.volume)
        self.hull = hull

    def sample(self):
        d = self.hull.area
        sections = [np.hypot(self.vertices[i], self.vertices[i + 1]) for i in range(len(self.vertices) - 1)]
        sections = [sections[i] + sections[i - 1] for i in range(1, len(sections))]
        contour_funcs = self.contour_func(sections, self.vertices)
        x = np.random.uniform(0, d)
        for i in range(len(sections)):
            if x < sections[i]:
                return contour_funcs[i](x)
        return self.vertices[0]

    def contour(self, n: int):
        d = self.hull.area
        sections = [np.hypot((self.vertices[i+1][0]-self.vertices[i][0]), (self.vertices[i+1][1]-self.vertices[i][1])) for i in range(len(self.vertices)-1)]
        for i in range(1,len(sections)):
            sections[i] += sections[i-1]

        xs = np.linspace(0, d, n)
        contour_funcs = self.contour_func(sections, self.vertices)
        ys = []
        i = 0
        for x in xs:
            while x > sections[i]:
                i += 1
            ys.append(contour_funcs[sections[i]](x))
        return np.array([xs, ys])

    def area(self) -> np.float32:
        if self.ar is not None:
            return np.float32(self.ar)
        else:
            ass = Assignment5()
            self.ar = ass.calculate_area_from_vertices(self.vertices)
            return self.ar

    def contour_func(self, sections, vertices):
        return {sections[i]: self.linear_func(vertices[i], vertices[i+1]) for i in range(len(sections))}

    def linear_func(self, p1, p2):
        m = (p2[1] - p1[1])/(p2[0] - p1[0])
        c = (p2[1] - (m*p2[0]))
        return lambda x: m * x + c


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def area(self, contour: callable, maxerr=0.001, n_points=400) -> np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        # n_points = 400
        vertices = contour(n_points)

        plt.scatter(vertices.T[0], vertices.T[1], c='r')
        plt.plot(vertices.T[0], vertices.T[1], c='black')
        plt.show()
        return self.calculate_area_from_vertices(vertices)

    def calculate_area_from_vertices(self, vertices):
        vertices.T[0] += abs(np.min(vertices.T[0]))
        vertices.T[1] += abs(np.min(vertices.T[1]))
        x =vertices.T[0]
        y = vertices.T[1]
        area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
        return area

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """

        T = time.time()
        x = sample()
        delta = time.time() - T
        if delta == 0:
            n = 1500
        else:
            n = int(maxtime*5/delta)
        if n == 0:
            shape = MyShape([x], ConvexHull([x]))
            return shape

        if n < 100:
            with ThreadPoolExecutor() as executor:
                samples = [executor.submit(sample) for _ in range(n)]
            samples = [s.result() for s in samples]
            samples += [samples[-1]]
            samples = np.array(samples)
            return MyShape(samples, ConvexHull(samples))
        scans = self.dbscan_get_points(n, sample)
        if len(scans) < 200:
            return MyShape(scans, ConvexHull(scans))
        amount = min(3, max(1, int(maxtime/3)))

        centers1 = self.kmeans_layer(scans, 200, amount)
        centers2 = self.kmeans_layer(centers1, 50, 20)
        centers3 = self.kmeans_layer(centers2, 25, 30)
        centers4 = self.kmeans_layer(centers3, 20, 5)
        hull = ConvexHull(centers4)
        vertices = [centers4[v] for v in hull.vertices]
        vertices = np.array(vertices + [vertices[0]])
        d_area = self.delauny_triangulation_points(centers4)
        print(f'finished at {time.time()-T} sec')

        xs = vertices.T[0]
        ys = vertices.T[1]
        plt.scatter(scans.T[0], scans.T[1], c='r')
        plt.scatter(centers1.T[0], centers1.T[1], c='g')
        plt.scatter(centers2.T[0], centers2.T[1], c='b')
        plt.scatter(centers3.T[0], centers3.T[1], c='orange')
        plt.plot(xs, ys, c='black')
        plt.show()

        # plt.plot(xs, ys, c='black')
        # plt.show()

        shape = MyShape(vertices, hull)
        # shape.ar = d_area

        return shape

    def kmeans_layer(self, X, clusters, num):
        centers = []
        for _ in range(num):
            kmeans = KMeans(n_clusters=clusters).fit(X)
            centers.append(kmeans.cluster_centers_)
        return np.concatenate(centers)

    def dbscan_get_points(self, n, f):
        scans = []
        for i in range(3):
            with ThreadPoolExecutor() as executor:
                samples = [executor.submit(f) for _ in range(n)]
            samples = [s.result() for s in samples]
            samples += [samples[-1]]
            samples = np.array(samples)
            dists = distance.cdist(samples, samples, "euclidean")
            eps = (np.max(samples.T[0]) - np.min(samples.T[0]))/20
            # eps = np.mean(dists)/np.max(dists)/6
            scan = DBSCAN(eps, min_samples=20).fit(samples)
            mapper = scan.labels_ != -1
            scans += list(samples[mapper])

            groups = np.unique(scan.labels_)
            fig, ax = plt.subplots()
            ax.margins(0.05)
            for group in groups:
                ax.plot(samples[scan.labels_ == group].T[0], samples[scan.labels_ == group].T[1], marker='.', linestyle='', ms=12)
            plt.show()
        return np.array(scans)

    def delauny_triangulation_points(self, points):
        max_dist = np.max(distance.cdist(points, points, 'euclidean'))
        delauny = Delaunay(points)
        area = 0
        for tri in delauny.simplices:
            max_local_dist = np.max(distance.cdist(points[tri], points[tri], 'euclidean'))
            if max_local_dist < max_dist / 2:
                area += self.calculate_area_from_vertices(points[tri])
        print(f"Delauny tri area: {area}")
        return area


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_area(self):
        ass5 = Assignment5()
        res = ass5.area(Circle(np.float32(1), np.float32(1), np.float32(3), np.float32(0)).contour)
        exp = np.pi*9
        print(f'calculated area is: {res}')
        print(f'relative error is: {abs(res-exp)/exp}')
        self.assertLessEqual(abs(res-exp)/exp, 0.001)

    def test_optimal_points(self):
        ass = Assignment5()
        ns = list(range(50, 10000, 50))
        errs = np.zeros([len(ns)])
        ts = np.zeros([len(ns)])
        for shape in [shape1(),shape2(),shape3(),shape4(),shape5(),shape6()]:
            print("shape started")
            expected = shape.area()
            for i in range(len(ns)):
                T = time.time()
                res = ass.area(shape.contour, n_points=ns[i])
                ts[i] += (time.time()-T)
                errs[i] += (abs(res-expected)/abs(expected))
        errs = errs/6
        ts = ts/6
        print(f'minimal error was in {ns[np.argmin(errs)]}')

        plt.plot(ns, errs)
        # plt.scatter(ns, errs, c='r')
        plt.show()
        plt.plot(ns, ts)
        plt.show()

    def test_fit_optimal_points(self):
        ass = Assignment5()
        ns = list(range(500, 10000, 500))
        errs = np.zeros([len(ns)])
        ts = np.zeros([len(ns)])
        # circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        # real_shape = Circle(np.float32(1), np.float32(1), np.float32(1), np.float32(0))
        shape = shape1()
        real_shape = shape1()
        expected = real_shape.area()
        for i in range(len(ns)):
            print(f"{ns[i]} points started started")
            for _ in range(5):
                T = time.time()
                res = ass.fit_shape(shape.sample, 30)
                res = res.area()
                ts[i] += (time.time() - T)
                errs[i] += (abs(res - expected) / abs(expected))
        errs = errs / 5
        ts = ts / 5
        print(f'minimal error was in {ns[np.argmin(errs)]}')

        plt.plot(ns, errs)
        # plt.scatter(ns, errs, c='r')
        plt.show()
        plt.plot(ns, ts)
        plt.show()


    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(0.05)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print(a)
        print(np.pi)
        self.assertLess(abs(a - np.pi)/(np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_fit_circle1(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print(a)
        circ = Circle(np.float32(1), np.float32(1), np.float32(1), np.float32(0))
        cont = circ.contour(400)
        plt.plot(shape.vertices.T[0], shape.vertices.T[1], c='r')
        plt.plot(cont.T[0], cont.T[1])
        plt.show()
        self.assertLess(abs(a - np.pi)/np.pi, 0.01)
        self.assertLessEqual(T, 32)


    def test_fit_circle2(self):
        circ = noisy_circle(cx=1, cy=1, radius=3, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print(a)
        circ = Circle(np.float32(1), np.float32(1), np.float32(3), np.float32(0))
        cont = circ.contour(400)
        plt.plot(shape.vertices.T[0], shape.vertices.T[1], c='r')
        plt.plot(cont.T[0], cont.T[1])
        plt.show()
        self.assertLess(abs(a - np.pi*9)/(np.pi*9), 0.01)
        self.assertLessEqual(T, 32)

    def test_fit_shape1(self):
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=shape1().sample, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print(a)
        cont = shape1().contour(400)
        plt.plot(shape.vertices.T[0], shape.vertices.T[1], c='r')
        plt.plot(cont.T[0], cont.T[1])
        plt.show()
        self.assertLess(abs(a - shape1().area()) / shape1().area(), 0.01)
        self.assertLessEqual(T, 32)

    def test_fit_shape2(self):
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=shape2().sample, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print(a)
        cont = shape2().contour(400)
        plt.plot(shape.vertices.T[0], shape.vertices.T[1], c='r')
        plt.plot(cont.T[0], cont.T[1])
        plt.show()
        self.assertLess(abs(a - shape2().area()) / shape2().area(), 0.01)
        self.assertLessEqual(T, 32)

    def test_fit_shape3(self):
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=shape3().sample, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print(a)
        cont = shape3().contour(400)
        plt.plot(shape.vertices.T[0], shape.vertices.T[1], c='r')
        plt.plot(cont.T[0], cont.T[1])
        plt.show()
        self.assertLess(abs(a - shape3().area()) / shape3().area(), 0.01)
        self.assertLessEqual(T, 32)

    def test_fit_shape4(self):
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=shape4().sample, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print(a)
        cont = shape4().contour(400)
        plt.plot(shape.vertices.T[0], shape.vertices.T[1], c='r')
        plt.plot(cont.T[0], cont.T[1])
        plt.show()
        self.assertLess(abs(a - shape4().area()) / shape4().area(), 0.01)
        self.assertLessEqual(T, 32)

    def test_fit_shape5(self):
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=shape5().sample, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print(a)
        cont = shape5().contour(400)
        plt.plot(shape.vertices.T[0], shape.vertices.T[1], c='r')
        plt.plot(cont.T[0], cont.T[1])
        plt.show()
        self.assertLess(abs(a - shape5().area()) / shape5().area(), 0.01)
        self.assertLessEqual(T, 32)

    def test_contour(self):
        circ = noisy_circle(cx=1, cy=1, radius=3, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        print(T)
        points = shape.contour(400)
        plt.plot(points.T[0], points.T[1], c='r')
        circ = Circle(np.float32(1), np.float32(1), np.float32(3), np.float32(0))
        cont = circ.contour(400)
        plt.plot(cont.T[0], cont.T[1])
        plt.show()


if __name__ == "__main__":
    unittest.main()
