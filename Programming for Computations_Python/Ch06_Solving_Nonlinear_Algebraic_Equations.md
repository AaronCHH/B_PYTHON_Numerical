
# Ch06 Solving Nonlinear Algebraic Equations
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Ch06 Solving Nonlinear Algebraic Equations](#ch06-solving-nonlinear-algebraic-equations)
  * [6.1 Brute Force Methods](#61-brute-force-methods)
    * [6.1.1 Brute Force Root Finding](#611-brute-force-root-finding)
    * [6.1.2 Brute Force Optimization](#612-brute-force-optimization)
    * [6.1.3 Model Problem for Algebraic Equations](#613-model-problem-for-algebraic-equations)
  * [6.2 Newton’s Method](#62-newtons-method)
    * [6.2.1 Deriving and Implementing Newton’s Method](#621-deriving-and-implementing-newtons-method)
    * [6.2.2 Making a More Efficient and Robust Implementation](#622-making-a-more-efficient-and-robust-implementation)
  * [6.3 The Secant Method](#63-the-secant-method)
  * [6.4 The Bisection Method](#64-the-bisection-method)
  * [6.5 Rate of Convergence](#65-rate-of-convergence)
  * [6.6 Solving Multiple Nonlinear Algebraic Equations](#66-solving-multiple-nonlinear-algebraic-equations)
    * [6.6.1 Abstract Notation](#661-abstract-notation)
    * [6.6.2 Taylor Expansions for Multi-Variable Functions](#662-taylor-expansions-for-multi-variable-functions)
    * [6.6.3 Newton’s Method](#663-newtons-method)
    * [6.6.4 Implementation](#664-implementation)
  * [6.7 Exercises](#67-exercises)

<!-- tocstop -->


## 6.1 Brute Force Methods

### 6.1.1 Brute Force Root Finding


```python
# %load py/brute_force_root_finder_flat.py
from numpy import linspace, exp, cos

def f(x):
    return exp(-x**2)*cos(4*x)

x = linspace(0, 4, 10001)
y = f(x)

root = None  # Initialization
for i in range(len(x)-1):
    if y[i]*y[i+1] < 0:
         root = x[i] - (x[i+1] - x[i])/(y[i+1] - y[i])*y[i]
         break  # Jump out of loop

if root is None:
    print 'Could not find any root in [%g, %g]' % (x[0], x[-1])
else:
    print 'Find (the first) root as x=%g' % root

```

    Find (the first) root as x=0.392699



```python
# %load py/brute_force_root_finder_function.py
def brute_force_root_finder(f, a, b, n):
    from numpy import linspace
    x = linspace(a, b, n)
    y = f(x)
    roots = []
    for i in range(n-1):
        if y[i]*y[i+1] < 0:
            root = x[i] - (x[i+1] - x[i])/(y[i+1] - y[i])*y[i]
            roots.append(root)
    return roots

def demo():
    from numpy import exp, cos
    roots = brute_force_root_finder(
        lambda x: exp(-x**2)*cos(4*x), 0, 4, 1001)
    if roots:
        print roots
    else:
        print 'Could not find any roots'

if __name__ == '__main__':
    demo()

```

### 6.1.2 Brute Force Optimization


```python
# %load py/brute_force_optimizer.py
def brute_force_optimizer(f, a, b, n):
    from numpy import linspace
    x = linspace(a, b, n)
    y = f(x)
    # Let maxima and minima hold the indices corresponding
    # to (local) maxima and minima points
    minima = []
    maxima = []
    for i in range(n-1):
        if y[i-1] < y[i] > y[i+1]:
            maxima.append(i)
        if y[i-1] > y[i] < y[i+1]:
            minima.append(i)

    # What about the end points?
    y_max_inner = max([y[i] for i in maxima])
    y_min_inner = min([y[i] for i in minima])
    if y[0] > y_max_inner:
        maxima.append(0)
    if y[len(x)-1] > y_max_inner:
        maxima.append(len(x)-1)
    if y[0] < y_min_inner:
        minima.append(0)
    if y[len(x)-1] < y_min_inner:
        minima.append(len(x)-1)

    # Return x and y values
    return [(x[i], y[i]) for i in minima], \
           [(x[i], y[i]) for i in maxima]

def demo():
    from numpy import exp, cos
    minima, maxima = brute_force_optimizer(
        lambda x: exp(-x**2)*cos(4*x), 0, 4, 1001)
    print 'Minima:', minima
    print 'Maxima:', maxima

if __name__ == '__main__':
    demo()

```

### 6.1.3 Model Problem for Algebraic Equations

## 6.2 Newton’s Method

### 6.2.1 Deriving and Implementing Newton’s Method

### 6.2.2 Making a More Efficient and Robust Implementation


```python
# %load py/Newtons_method.py
def Newton(f, dfdx, x, eps):
    f_value = f(x)
    iteration_counter = 0
    while abs(f_value) > eps and iteration_counter < 100:
        try:
            x = x - float(f_value)/dfdx(x)
        except ZeroDivisionError:
            print "Error! - derivative zero for x = ", x
            sys.exit(1)     # Abort with error

        f_value = f(x)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(f_value) > eps:
        iteration_counter = -1
    return x, iteration_counter

def f(x):
    return x**2 - 9

def dfdx(x):
    return 2*x

solution, no_iterations = Newton(f, dfdx, x=1000, eps=1.0e-6)

if no_iterations > 0:    # Solution found
    print "Number of function calls: %d" % (1 + 2*no_iterations)
    print "A solution is: %f" % (solution)
else:
    print "Solution not found!"


```

## 6.3 The Secant Method


```python
# %load py/secant_method.py
def secant(f, x0, x1, eps):
    f_x0 = f(x0)
    f_x1 = f(x1)
    iteration_counter = 0
    while abs(f_x1) > eps and iteration_counter < 100:
        try:
            denominator = float(f_x1 - f_x0)/(x1 - x0)
            x = x1 - float(f_x1)/denominator
        except ZeroDivisionError:
            print "Error! - denominator zero for x = ", x
            sys. exit(1) # Abort with error
        x0 = x1
        x1 = x
        f_x0 = f_x1
        f_x1 = f(x1)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(f_x1) > eps:
        iteration_counter = -1
    return x, iteration_counter

def f(x):
    return x**2 - 9

x0 = 1000; x1 = x0 - 1

solution, no_iterations = secant(f, x0, x1, eps=1.0e-6)

if no_iterations > 0: # Solution found
    print "Number of function calls: %d" % (2 + no_iterations)
    print "A solution is: %f" % (solution)
else:
    print "Solution not found!"
```

## 6.4 The Bisection Method


```python
# %load py/bisection_method.py
def bisection(f, x_L, x_R, eps, return_x_list=False):
    f_L = f(x_L)
    if f_L*f(x_R) > 0:
        print "Error! Function does not have opposite \
                 signs at interval endpoints!"
        sys.exit(1)
    x_M = float(x_L + x_R)/2.0
    f_M = f(x_M)
    iteration_counter = 1
    if return_x_list:
        x_list = []

    while abs(f_M) > eps:
        if f_L*f_M > 0:   # i.e. same sign
            x_L = x_M
            f_L = f_M
        else:
            x_R = x_M
        x_M = float(x_L + x_R)/2
        f_M = f(x_M)
        iteration_counter += 1
        if return_x_list:
            x_list.append(x_M)
    if return_x_list:
        return x_list, iteration_counter
    else:
        return x_M, iteration_counter

def f(x):
    return x**2 - 9

a = 0;   b = 1000

solution, no_iterations = bisection(f, a, b, eps=1.0e-6)

print "Number of function calls: %d" % (1 + 2*no_iterations)
print "A solution is: %f" % (solution)

```

## 6.5 Rate of Convergence


```python
# %load py/nonlinear_solvers.py
import sys

def bisection(f, x_L, x_R, eps, return_x_list=False):
    f_L = f(x_L)
    if f_L*f(x_R) > 0:
        print "Error! Function does not have opposite \
                 signs at interval endpoints!"
        sys.exit(1)
    x_M = float(x_L + x_R)/2
    f_M = f(x_M)
    iteration_counter = 1
    if return_x_list:
        x_list = []

    while abs(f_M) > eps:
        if f_L*f_M > 0:   # i.e., same sign
            x_L = x_M
            f_L = f_M
        else:
            x_R = x_M
        x_M = float(x_L + x_R)/2
        f_M = f(x_M)
        iteration_counter += 1
        if return_x_list:
            x_list.append(x_M)
    if return_x_list:
        return x_list, iteration_counter
    else:
        return x_M, iteration_counter

def Newton(f, dfdx, x, eps, return_x_list=False):
    f_value = f(x)
    iteration_counter = 0
    if return_x_list:
        x_list = []

    while abs(f_value) > eps and iteration_counter < 100:
        try:
            x = x - float(f_value)/dfdx(x)
        except ZeroDivisionError:
            print "Error! - derivative zero for x = ", x
            sys.exit(1)     # Abort with error

        f_value = f(x)
        iteration_counter += 1
        if return_x_list:
            x_list.append(x)

    # Here, either a solution is found, or too many iterations
    if abs(f_value) > eps:
        iteration_counter = -1  # i.e., lack of convergence

    if return_x_list:
        return x_list, iteration_counter
    else:
        return x, iteration_counter

def secant(f, x0, x1, eps, return_x_list=False):
    f_x0 = f(x0)
    f_x1 = f(x1)
    iteration_counter = 0
    if return_x_list:
        x_list = []

    while abs(f_x1) > eps and iteration_counter < 100:
        try:
            denominator = float(f_x1 - f_x0)/(x1 - x0)
            x = x1 - float(f_x1)/denominator
        except ZeroDivisionError:
            print "Error! - denominator zero for x = ", x
            sys.exit(1)     # Abort with error
        x0 = x1
        x1 = x
        f_x0 = f_x1
        f_x1 = f(x1)
        iteration_counter += 1
        if return_x_list:
            x_list.append(x)
    # Here, either a solution is found, or too many iterations
    if abs(f_x1) > eps:
        iteration_counter = -1

    if return_x_list:
        return x_list, iteration_counter
    else:
        return x, iteration_counter

from math import log

def rate(x, x_exact):
    e = [abs(x_ - x_exact) for x_ in x]
    q = [log(e[n+1]/e[n])/log(e[n]/e[n-1])
         for n in range(1, len(e)-1, 1)]
    return q

```

## 6.6 Solving Multiple Nonlinear Algebraic Equations

### 6.6.1 Abstract Notation

### 6.6.2 Taylor Expansions for Multi-Variable Functions


```python
from sympy import *
x0, x1 = symbols('x0 x1' )
F0 = x0**2 - x1 + x0*cos(pi*x0)
F1 = x0*x1 + exp(-x1) - x0**(-1)
diff(F0, x0)
diff(F0, x1)
diff(F1, x0)
diff(F1, x1)
```




    x0 - exp(-x1)



### 6.6.3 Newton’s Method

### 6.6.4 Implementation

## 6.7 Exercises

* Exercise 6.1: Understand why Newton's method can fail
* Exercise 6.2: See if the secant method fails
* Exercise 6.4: Combine the bisection method with Newton's method
* Exercise 6.5: Write a test function for Newton's method
* Exercise 6.6: Solve nonlinear equation for a vibrating beam


```python

```
