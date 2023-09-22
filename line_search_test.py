import scipy.optimize as opt  
import numpy as np 



def f(x):
    return x**2 + 12

def grad(x):
    return 2*x
a = opt.line_search(f, grad, 0.5, -grad(0.5))
print(a[0])




def f(x):
    return ((300)/ (200 + x))**2

def grad(x):
     ep =  0.00001
     xh = x + 0.00001
     return (f(xh) - f(x)) /  0.00001
a = opt.line_search(f, grad, 0.5, -grad(0.5))
print(a[0])
a = opt.line_search(f, grad, -.5, -grad(-0.5))
print(a[0])
