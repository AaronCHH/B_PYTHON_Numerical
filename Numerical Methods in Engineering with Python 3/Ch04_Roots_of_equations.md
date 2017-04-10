
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

- [4 Roots of Equations](#4-roots-of-equations)
- [4.1 Introduction](#41-introduction)
- [4.2 Incremental Search Method](#42-incremental-search-method)
- [4.3 Method of Bisection](#43-method-of-bisection)
	- [EXAMPLE4.2](#example42)
		- [EXAMPLE4.2](#example42-1)
	- [EXAMPLE4.3](#example43)
		- [EXAMPLE4.3](#example43-1)
- [4.4 Methods Based on Linear Interpolation](#44-methods-based-on-linear-interpolation)
	- [Secant and False Position Methods](#secant-and-false-position-methods)
	- [Ridder’s Method](#ridders-method)
		- [EXAMPLE4.5](#example45)
- [4.5 Newton-Raphson Method](#45-newton-raphson-method)
	- [EXAMPLE4.8](#example48)
		- [EXAMPLE4.8](#example48-1)
- [4.6 Systems of Equations](#46-systems-of-equations)
	- [Newton-Raphson Method](#newton-raphson-method)
		- [EXAMPLE4.9](#example49)
- [4.7 Zeros of Polynomials](#47-zeros-of-polynomials)
	- [Introduction](#introduction)
	- [Evaluation of Polynomials](#evaluation-of-polynomials)
	- [Deflation of Polynomials](#deflation-of-polynomials)
	- [Laguerre’s Method](#laguerres-method)
		- [EXAMPLE4.12](#example412)
- [4.8 Other Methods](#48-other-methods)

<!-- tocstop -->

# 4 Roots of Equations

# 4.1 Introduction

# 4.2 Incremental Search Method


```python
# %load code/rootsearch.py
## module rootsearch
''' x1,x2 = rootsearch(f,a,b,dx).
    Searches the interval (a,b) in increments dx for
    the bounds (x1,x2) of the smallest root of f(x).
    Returns x1 = x2 = None if no roots were detected.
'''
from numpy import sign

def rootsearch(f,a,b,dx):
    x1 = a; f1 = f(a)
    x2 = a + dx; f2 = f(x2)
    while sign(f1) == sign(f2):
        if x1  >=  b: return None,None
        x1 = x2; f1 = f2
        x2 = x1 + dx; f2 = f(x2)
    else:
        return x1,x2

```

# 4.3 Method of Bisection


```python
# %load code/bisection.py
## module bisection
''' root = bisection(f,x1,x2,switch=0,tol=1.0e-9).
    Finds a root of f(x) = 0 by bisection.
    The root must be bracketed in (x1,x2).
    Setting switch = 1 returns root = None if
    f(x) increases upon bisection.
'''    
import math
import error
from numpy import sign

def bisection(f,x1,x2,switch=1,tol=1.0e-9):
    f1 = f(x1)
    if f1 == 0.0: return x1
    f2 = f(x2)
    if f2 == 0.0: return x2
    if sign(f1) == sign(f2):
        error.err('Root is not bracketed')
    n = int(math.ceil(math.log(abs(x2 - x1)/tol)/math.log(2.0)))

    for i in range(n):
        x3 = 0.5*(x1 + x2); f3 = f(x3)
        if (switch == 1) and (abs(f3) > abs(f1)) \
                         and (abs(f3) > abs(f2)):
            return None   
        if f3 == 0.0: return x3
        if sign(f2)!= sign(f3): x1 = x3; f1 = f3
        else: x2 = x3; f2 = f3
    return (x1 + x2)/2.0

```

## EXAMPLE4.2
### EXAMPLE4.2


```python
#!/usr/bin/python
## example4_2
from bisection import *
def f(x): return x**3 - 10.0*x**2 + 5.0
x = bisection(f, 0.0, 1.0, tol = 1.0e-4)
print('x =', '{:6.4f}'.format(x))
input("Press return to exit")
```

    x = 0.7346
    Press return to exit





    ''



## EXAMPLE4.3
### EXAMPLE4.3


```python
#!/usr/bin/python
## example4_3
import math
from rootsearch import *
from bisection import *
def f(x): return x - math.tan(x)
a,b,dx = (0.0, 20.0, 0.01)
print("The roots are:")
while True:
    x1,x2 = rootsearch(f,a,b,dx)
    if x1 != None:
        a = x2
        root = bisection(f,x1,x2,1)
        if root != None: print(root)
    else:
        print("\nDone")
        break
input("Press return to exit")
```

    The roots are:
    0.0
    4.493409458100745
    7.725251837074637
    10.904121659695917
    14.06619391292308
    17.220755272209537

    Done
    Press return to exit





    ''



# 4.4 Methods Based on Linear Interpolation

## Secant and False Position Methods

## Ridder’s Method


```python
# %load code/ridder.py
## module ridder
''' root = ridder(f,a,b,tol=1.0e-9).
    Finds a root of f(x) = 0 with Ridder's method.
    The root must be bracketed in (a,b).
'''
import error
import math
from numpy import sign

def ridder(f,a,b,tol=1.0e-9):   
    fa = f(a)
    if fa == 0.0: return a
    fb = f(b)
    if fb == 0.0: return b
    if sign(fa) == sign(fb): error.err('Root is not bracketed')
    for i in range(30):
      # Compute the improved root x from Ridder's formula
        c = 0.5*(a + b); fc = f(c)
        s = math.sqrt(fc**2 - fa*fb)
        if s == 0.0: return None
        dx = (c - a)*fc/s
        if (fa - fb) < 0.0: dx = -dx
        x = c + dx; fx = f(x)
      # Test for convergence
        if i > 0:
            if abs(x - xOld) < tol*max(abs(x),1.0): return x
        xOld = x
      # Re-bracket the root as tightly as possible
        if sign(fc) == sign(fx):
            if sign(fa)!= sign(fx): b = x; fb = fx
            else: a = x; fa = fx
        else:
            a = c; b = x; fa = fc; fb = fx
    return None
    print('Too many iterations')


```

### EXAMPLE4.5


```python
#!/usr/bin/python
## example4_5
from ridder import *
def f(x):
    a = (x - 0.3)**2 + 0.01
    b = (x - 0.8)**2 + 0.04
    return 1.0/a - 1.0/b
print("root =",ridder(f,0.0,1.0))
input("Press return to exit")
```

    root = 0.5800000000000001
    Press return to exit





    ''



# 4.5 Newton-Raphson Method


```python
# %load code/newtonRaphson.py
## module newtonRaphson
''' root = newtonRaphson(f,df,a,b,tol=1.0e-9).
    Finds a root of f(x) = 0 by combining the Newton-Raphson
    method with bisection. The root must be bracketed in (a,b).
    Calls user-supplied functions f(x) and its derivative df(x).   
'''    
def newtonRaphson(f,df,a,b,tol=1.0e-9):
    import error
    from numpy import sign

    fa = f(a)
    if fa == 0.0: return a
    fb = f(b)
    if fb == 0.0: return b
    if sign(fa) == sign(fb): error.err('Root is not bracketed')
    x = 0.5*(a + b)                    
    for i in range(30):
        fx = f(x)
        if fx == 0.0: return x
      # Tighten the brackets on the root
        if sign(fa) != sign(fx): b = x  
        else: a = x
      # Try a Newton-Raphson step    
        dfx = df(x)
      # If division by zero, push x out of bounds
        try: dx = -fx/dfx
        except ZeroDivisionError: dx = b - a
        x = x + dx
      # If the result is outside the brackets, use bisection  
        if (b - x)*(x - a) < 0.0:  
            dx = 0.5*(b - a)                      
            x = a + dx
      # Check for convergence     
        if abs(dx) < tol*max(abs(b),1.0): return x
    print('Too many iterations in Newton-Raphson')

```

## EXAMPLE4.8
### EXAMPLE4.8

# 4.6 Systems of Equations

## Newton-Raphson Method


```python
# %load code/newtonRaphson2.py
## module newtonRaphson2
''' soln = newtonRaphson2(f,x,tol=1.0e-9).
    Solves the simultaneous equations f(x) = 0 by
    the Newton-Raphson method using {x} as the initial
    guess. Note that {f} and {x} are vectors.
'''
import numpy as np
from gaussPivot import *
import math
def newtonRaphson2(f,x,tol=1.0e-9):

    def jacobian(f,x):
        h = 1.0e-4
        n = len(x)
        jac = np.zeros((n,n))
        f0 = f(x)
        for i in range(n):
            temp = x[i]
            x[i] = temp + h
            f1 = f(x)
            x[i] = temp
            jac[:,i] = (f1 - f0)/h
        return jac,f0

    for i in range(30):
        jac,f0 = jacobian(f,x)
        if math.sqrt(np.dot(f0,f0)/len(x)) < tol:
            return x
        dx = gaussPivot(jac,-f0)
        x = x + dx
        if math.sqrt(np.dot(dx,dx)) < tol*max(max(abs(x)),1.0): return x
    print('Too many iterations')

```

### EXAMPLE4.9


```python
#!/usr/bin/python
## example4_10
import numpy as np
import math
from newtonRaphson2 import *
def f(x):
    f = np.zeros(len(x))
    f[0] = math.sin(x[0]) + x[1]**2 + math.log(x[2]) - 7.0
    f[1] = 3.0*x[0] + 2.0**x[1] - x[2]**3 + 1.0
    f[2] = x[0] + x[1] + x[2] - 5.0
    return f
```

# 4.7 Zeros of Polynomials

## Introduction

## Evaluation of Polynomials

## Deflation of Polynomials

## Laguerre’s Method


```python
# %load code/polyRoots.py
## module polyRoots
''' roots = polyRoots(a).
    Uses Laguerre's method to compute all the roots of
    a[0] + a[1]*x + a[2]*x^2 +...+ a[n]*x^n = 0.
    The roots are returned in the array 'roots',
'''    
from evalPoly import *
import numpy as np
import cmath
from random import random

def polyRoots(a,tol=1.0e-12):

    def laguerre(a,tol):
        x = random()   # Starting value (random number)
        n = len(a) - 1
        for i in range(30):
            p,dp,ddp = evalPoly(a,x)
            if abs(p) < tol: return x
            g = dp/p
            h = g*g - ddp/p
            f = cmath.sqrt((n - 1)*(n*h - g*g))
            if abs(g + f) > abs(g - f): dx = n/(g + f)
            else: dx = n/(g - f)
            x = x - dx
            if abs(dx) < tol: return x
        print('Too many iterations')

    def deflPoly(a,root):  # Deflates a polynomial
        n = len(a)-1
        b = [(0.0 + 0.0j)]*n
        b[n-1] = a[n]
        for i in range(n-2,-1,-1):
            b[i] = a[i+1] + root*b[i+1]
        return b

    n = len(a) - 1
    roots = np.zeros((n),dtype=complex)
    for i in range(n):
        x = laguerre(a,tol)
        if abs(x.imag) < tol: x = x.real
        roots[i] = x
        a = deflPoly(a,x)
    return roots

```

### EXAMPLE4.12


```python
#!/usr/bin/python
## example4_12
from polyRoots import *
import numpy as np
c = np.array([-250.0,155.0,-9.0,-5.0,1.0])
print('Roots are:\n',polyRoots(c))
input('Press return to exit')
```

    Roots are:
     [ 2.+0.j  4.-3.j  4.+3.j -5.+0.j]
    Press return to exit





    ''



# 4.8 Other Methods
