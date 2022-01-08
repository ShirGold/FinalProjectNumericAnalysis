import numpy as np


def function1(x):
    return np.polynomial.Polynomial([5])(x)
def function2(x):
    return np.polynomial.Polynomial([5,-3,1])(x)
def function3(x):
    return np.sin(x**2)
def function4(x):
    return np.exp(-2*(x**2))
def function5(x):
    return np.sin(x)/x
def function6(x):
    return 1/np.log(x)
def function7(x):
    return np.exp(np.exp(x))
def function8(x):
    return np.log(np.log(x))
def function9(x):
    return np.sin(np.log(x))
def function10(x):
    return np.arctan(x)
def function11(x):
    return np.power(2, (1/(x**2)))*np.sin(1/x)

def get_all_functions():
    fs = {
        'f(x)= 5': function1,
        'f(x)= x^2 - 3x + 5': function2,
        'f(x)= sin(x^2)': function3,
        'f(x)= e^(-2x^2)': function4,
        'f(x)= arctan(x)': function10,
        'f(x)= sin(x)/x': function5,
        'f(x)= 1/ln(x)': function6,
        'f(x)= e^(e^x)': function7,
        'f(x)= ln(ln(x))': function8,
        'f(x)= sin(ln(x))': function9,
        'f(x)= 2^(1/x^2)*sin(1/x)': function11
    }
    return fs
