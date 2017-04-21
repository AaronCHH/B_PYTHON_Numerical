
# 3 Computing Integrals
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [3 Computing Integrals](#3-computing-integrals)
  * [3.1 Basic Ideas of Numerical Integration](#31-basic-ideas-of-numerical-integration)
  * [3.2 The Composite Trapezoidal Rule](#32-the-composite-trapezoidal-rule)
    * [3.2.1 The General Formula](#321-the-general-formula)
    * [3.2.2 Implementation](#322-implementation)
    * [3.2.3 Making a Module](#323-making-a-module)
    * [3.2.4 Alternative Flat Special-Purpose Implementation](#324-alternative-flat-special-purpose-implementation)
  * [3.3 The Composite Midpoint Method](#33-the-composite-midpoint-method)
    * [3.3.1 The General Formula](#331-the-general-formula)
    * [3.3.2 Implementation](#332-implementation)
    * [3.3.3 Comparing the Trapezoidal and the Midpoint Methods](#333-comparing-the-trapezoidal-and-the-midpoint-methods)
  * [3.4 Testing](#34-testing)
    * [3.4.1 Problems with Brief Testing Procedures](#341-problems-with-brief-testing-procedures)
    * [3.4.2 Proper Test Procedures](#342-proper-test-procedures)
    * [3.4.3 Finite Precision of Floating-Point Numbers](#343-finite-precision-of-floating-point-numbers)
    * [3.4.4 Constructing Unit Tests and Writing Test Functions](#344-constructing-unit-tests-and-writing-test-functions)
  * [3.5 Vectorization](#35-vectorization)
  * [3.6 Measuring Computational Speed](#36-measuring-computational-speed)
  * [3.7 Double and Triple Integrals](#37-double-and-triple-integrals)
    * [3.7.1 The Midpoint Rule for a Double Integral](#371-the-midpoint-rule-for-a-double-integral)
    * [3.7.2 The Midpoint Rule for a Triple Integral](#372-the-midpoint-rule-for-a-triple-integral)
    * [3.7.3 Monte Carlo Integration for Complex-Shaped Domains](#373-monte-carlo-integration-for-complex-shaped-domains)
  * [3.8 Exercises](#38-exercises)

<!-- tocstop -->


## 3.1 Basic Ideas of Numerical Integration

## 3.2 The Composite Trapezoidal Rule

### 3.2.1 The General Formula

### 3.2.2 Implementation


```python
# %load py/trapezoidal.py
def trapezoidal(f, a, b, n):
    h = float(b-a)/float(n)
    result = 0.5*f(a) + 0.5*f(b)
    for i in range(1, n):
        result += f(a + i*h)
    result *= h
    return result

def application():
    from math import exp
    v = lambda t: 3*(t**2)*exp(t**3)
    n = input('n: ')
    numerical = trapezoidal(v, 0, 1, n)

    # Compare with exact result
    V = lambda t: exp(t**3)
    exact = V(1) - V(0)
    error = exact - numerical
    print('n=%d: %.16f, error: %g' % (n, numerical, error))

if __name__ == '__main__':
    application()

```


```python
%run py/trapezoidal.py
```


```python
from math import exp
a = 0.0; b = 1.0
n = int(input('n: ' ))
dt = float(b - a)/n

# Integral by the trapezoidal method
numerical = 0.5*3*(a**2)*exp(a**3) + 0.5*3*(b**2)*exp(b**3)

for i in range(1, n):
    numerical += 3*((a + i*dt)**2)*exp((a + i*dt)**3)
    numerical *= dt
    exact_value = exp(1**3) - exp(0**3)
    error = abs(exact_value - numerical)
    rel_error = (error/exact_value)*100
    print('n=%d: %.16f, error: %g' % (n, numerical, error))
```

    n: 4
    n=4: 1.0669688595121429, error: 0.651313
    n=4: 0.4792075498280657, error: 1.23907
    n=4: 0.7630844434624198, error: 0.955197



```python
from math import exp
v = lambda t: 3*(t**2)*exp(t**3) # Function to be integrated
a = 0.0; b = 1.0
n = int(input('n: ' ))
dt = float(b - a)/n

# Integral by the trapezoidal method
numerical = 0.5*v(a) + 0.5*v(b)
for i in range(1, n):
    numerical += v(a + i*dt)
numerical *= dt

F = lambda t: exp(t**3)
exact_value = F(b) - F(a)
error = abs(exact_value - numerical)
rel_error = (error/exact_value)*100
print('n=%d: %.16f, error: %g' % (n, numerical, error))
```

    n: 4
    n=4: 1.9227167504675762, error: 0.204435



```python
from trapezoidal import trapezoidal
from math import exp
trapezoidal(lambda x: exp(-x**2), -1, 1.1, 400)
```

### 3.2.3 Making a Module

### 3.2.4 Alternative Flat Special-Purpose Implementation

## 3.3 The Composite Midpoint Method

### 3.3.1 The General Formula

### 3.3.2 Implementation


```python
def midpoint(f, a, b, n):
    h = float(b-a)/n
    result = 0
    for i in range(n):
        result += f((a + h/2.0) + i*h)
    result *= h
    return result
```

### 3.3.3 Comparing the Trapezoidal and the Midpoint Methods


```python
import sys
sys.path.append("./py")
```


```python
# %load py/compare_integration_methods.py
from trapezoidal import trapezoidal
from midpoint import midpoint
from math import exp

g = lambda y: exp(-y**2)
a = 0
b = 2
print '    n        midpoint          trapezoidal'
for i in range(1, 21):
    n = 2**i
    m = midpoint(g, a, b, n)
    t = trapezoidal(g, a, b, n)
    print '%7d %.16f %.16f' % (n, m, t)

```

        n        midpoint          trapezoidal
          2 0.8842000076332692 0.8770372606158094
          4 0.8827889485397279 0.8806186341245393
          8 0.8822686991994210 0.8817037913321336
         16 0.8821288703366458 0.8819862452657772
         32 0.8820933014203766 0.8820575578012112
         64 0.8820843709743319 0.8820754296107942
        128 0.8820821359746071 0.8820799002925637
        256 0.8820815770754198 0.8820810181335849
        512 0.8820814373412922 0.8820812976045025
       1024 0.8820814024071774 0.8820813674728968
       2048 0.8820813936736116 0.8820813849400392
       4096 0.8820813914902204 0.8820813893068272
       8192 0.8820813909443684 0.8820813903985197
      16384 0.8820813908079066 0.8820813906714446
      32768 0.8820813907737911 0.8820813907396778
      65536 0.8820813907652575 0.8820813907567422
     131072 0.8820813907631487 0.8820813907610036
     262144 0.8820813907625702 0.8820813907620528
     524288 0.8820813907624605 0.8820813907623183
    1048576 0.8820813907624268 0.8820813907623890


## 3.4 Testing

### 3.4.1 Problems with Brief Testing Procedures

### 3.4.2 Proper Test Procedures

### 3.4.3 Finite Precision of Floating-Point Numbers


```python
a = 1;
b = 2;
expected = 3
a + b == expected
```




    True




```python
a = 0.1;
b = 0.2;
expected = 0.3
a + b == expected
```




    False




```python
print '%.17f\n%.17f\n%.17f\n%.17f' % (0.1, 0.2, 0.1 + 0.2, 0.3)
```

    0.10000000000000001
    0.20000000000000001
    0.30000000000000004
    0.29999999999999999



```python
a = 0.1;
b = 0.2;
expected = 0.3
computed = a + b
diff = abs(expected - computed)
tol = 1E-15
diff < tol
```




    True



### 3.4.4 Constructing Unit Tests and Writing Test Functions

Python has several frameworks for automatically running and checking a potentially very large number of tests for parts of your software by one command.
This is an extremely useful feature during program development: whenever you have done some changes to one or more files, launch the test command and make sure nothing is broken because of your edits.

The test frameworks nose and py.test are particularly attractive as they are very easy to use.
Tests are placed in special test functions that the frameworks can recognize and run for you.
The requirements to a test function are simple:   the name must start with test_

* the test function cannot have any arguments
* the tests inside test functions must be boolean expressions
* a boolean expression b must be tested with assert b, msg, where msg is an optional object (string or number) to be written out when b is false



```python
def add(a, b):
    return a + b
```


```python
def test_add():
    expected = 1 + 1
    computed = add(1, 1)
    assert computed == expected, '1+1=%g' % computed
```


```python
def test_add():
    expected = 0.3
    computed = add(0.1, 0.2)
    tol = 1E-14
    diff = abs(expected - computed)
    assert diff < tol, 'diff=%g' % diff
```


```python
# %load py/test_trapezoidal.py
from trapezoidal import trapezoidal

def test_trapezoidal_one_exact_result():
    """Compare one hand-computed result."""
    from math import exp
    v = lambda t: 3*(t**2)*exp(t**3)
    n = 2
    numerical = trapezoidal(v, 0, 1, n)
    exact = 2.463642041244344
    err = abs(exact - numerical)
    tol = 1E-14
    success = err < tol
    msg = 'error=%g > tol=%g' % (err, tol)
    assert success, msg

def test_trapezoidal_linear():
    """Check that linear functions are integrated exactly."""
    f = lambda x: 6*x - 4
    F = lambda x: 3*x**2 - 4*x  # Anti-derivative
    a = 1.2; b = 4.4
    exact = F(b) - F(a)
    tol = 1E-14
    for n in 2, 20, 21:
        numerical = trapezoidal(f, a, b, n)
        err = abs(exact - numerical)
        success = err < tol
        msg = 'n=%d, err=%g' % (n, err)
        assert success, msg

def convergence_rates(f, F, a, b, num_experiments=14):
    from math import log
    from numpy import zeros
    exact = F(b) - F(a)
    n = zeros(num_experiments, dtype=int)
    E = zeros(num_experiments)
    r = zeros(num_experiments-1)
    for i in range(num_experiments):
        n[i] = 2**(i+1)
        numerical = trapezoidal(f, a, b, n[i])
        E[i] = abs(exact - numerical)
        if i > 0:
            r_im1 = log(E[i]/E[i-1])/log(float(n[i])/n[i-1])
            r[i-1] = float('%.2f' % r_im1)  # Truncate to two decimals
    return r

def test_trapezoidal_conv_rate():
    """Check empirical convergence rates against the expected -2."""
    from math import exp
    v = lambda t: 3*(t**2)*exp(t**3)
    V = lambda t: exp(t**3)
    a = 1.1; b = 1.9
    r = convergence_rates(v, V, a, b, 14)
    print r
    tol = 0.01
    assert (abs(r[-1]) - 2) < tol, r[-4:]

```

## 3.5 Vectorization


```python
def f(x):
    return exp(-x)*sin(x) + 5*x

from numpy import exp, sin, linspace
x = linspace(0, 4, 101) # coordinates from 100 intervals on [0, 4]
y = f(x) # all points evaluated at once
```


```python
from numpy import linspace, sum
def midpoint(f, a, b, n):
    h = float(b-a)/n
    x = linspace(a + h/2, b - h/2, n)
    return h*sum(f(x))
```


```python
# %load py/integration_methods_vec.py
from numpy import linspace, sum

def midpoint(f, a, b, n):
    h = float(b-a)/n
    x = linspace(a + h/2, b - h/2, n)
    return h*sum(f(x))

def trapezoidal(f, a, b, n):
    h = float(b-a)/n
    x = linspace(a, b, n+1)
    s = sum(f(x)) - 0.5*f(a) - 0.5*f(b)
    return h*s

```


```python
from integration_methods_vec import midpoint
from numpy import exp

v = lambda t: 3*t**2*exp(t**3)
midpoint(v, 0, 1, 10)
```




    1.7014827690091872




```python
def trapezoidal(f, a, b, n):
    h = float(b-a)/n
    x = linspace(a, b, n+1)
    s = sum(f(x)) - 0.5*f(a) - 0.5*f(b)
return h*s
```

## 3.6 Measuring Computational Speed


```python
from integration_methods_vec import midpoint as midpoint_vec
from midpoint import midpoint
from numpy import exp
v = lambda t: 3*t**2*exp(t**3)
```


```python
%timeit midpoint_vec(v, 0, 1, 1000000)
```

    10 loops, best of 3: 100 ms per loop



```python
%timeit midpoint(v, 0, 1, 1000000)
```

    1 loop, best of 3: 1.47 s per loop



```python
8.17/(379*0.001) # efficiency f
```




    21.556728232189972



## 3.7 Double and Triple Integrals

### 3.7.1 The Midpoint Rule for a Double Integral


```python
def midpoint_double1(f, a, b, c, d, nx, ny):
    hx = (b - a)/float(nx)
    hy = (d - c)/float(ny)
    I = 0
    for i in range(nx):
        for j in range(ny):
            xi = a + hx/2 + i*hx
            yj = c + hy/2 + j *hy
            I += hx*hy*f(xi, yj )
    return I
```


```python
from midpoint_double import midpoint_double1

def f(x, y):
    return 2*x + y

midpoint_double1(f, 0, 2, 2, 3, 5, 5)
```




    9.000000000000005




```python
def midpoint(f, a, b, n):
    h = float(b-a)/n
    result = 0
    for i in range(n):
        result += f((a + h/2.0) + i*h)
    result *= h
    return result
```


```python
def midpoint_double2(f, a, b, c, d, nx, ny):
    def g(x):
        return midpoint(lambda y: f(x, y), c, d, ny)
    return midpoint(g, a, b, nx)
```


```python
def test_midpoint_double():
    """Test that a linear function is integrated exactly."""
    def f(x, y):
        return 2*x + y

    a = 0; b = 2; c = 2; d = 3
    import sympy
    x, y = sympy. symbols('x y')
    I_expected = sympy. integrate(f(x, y), (x, a, b), (y, c, d))
    # Test three cases: nx < ny, nx = ny, nx > ny
    for nx, ny in (3, 5), (4, 4), (5, 3):
        I_computed1 = midpoint_double1(f, a, b, c, d, nx, ny)
        I_computed2 = midpoint_double2(f, a, b, c, d, nx, ny)
        tol = 1E-14
        #print I_expected, I_computed1, I_computed2
        assert abs(I_computed1 - I_expected) < tol
        assert abs(I_computed2 - I_expected) < tol
```

### 3.7.2 The Midpoint Rule for a Triple Integral


```python
# %load py/midpoint_triple.py
def midpoint_triple1(g, a, b, c, d, e, f, nx, ny, nz):
    hx = (b - a)/float(nx)
    hy = (d - c)/float(ny)
    hz = (f - e)/float(nz)
    I = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                xi = a + hx/2 + i*hx
                yj = c + hy/2 + j*hy
                zk = e + hz/2 + k*hz
                I += hx*hy*hz*g(xi, yj, zk)
    return I

def midpoint(f, a, b, n):
    h = float(b-a)/n
    result = 0
    for i in range(n):
        result += f((a + h/2.0) + i*h)
    result *= h
    return result

def midpoint_triple2(g, a, b, c, d, e, f, nx, ny, nz):
    def p(x, y):
        return midpoint(lambda z: g(x, y, z), e, f, nz)

    def q(x):
        return midpoint(lambda y: p(x, y), c, d, ny)

    return midpoint(q, a, b, nx)

def test_midpoint_triple():
    """Test that a linear function is integrated exactly."""
    def g(x, y, z):
        return 2*x + y - 4*z

    a = 0;  b = 2;  c = 2;  d = 3;  e = -1;  f = 2
    import sympy
    x, y, z = sympy.symbols('x y z')
    I_expected = sympy.integrate(
        g(x, y, z), (x, a, b), (y, c, d), (z, e, f))
    for nx, ny, nz in (3, 5, 2), (4, 4, 4), (5, 3, 6):
        I_computed1 = midpoint_triple1(
            g, a, b, c, d, e, f, nx, ny, nz)
        I_computed2 = midpoint_triple2(
            g, a, b, c, d, e, f, nx, ny, nz)
        tol = 1E-14
        print I_expected, I_computed1, I_computed2
        assert abs(I_computed1 - I_expected) < tol
        assert abs(I_computed2 - I_expected) < tol

if __name__ == '__main__':
    test_midpoint_triple()

```

### 3.7.3 Monte Carlo Integration for Complex-Shaped Domains


```python
# %load py/MC_double.py
import numpy as np

def MonteCarlo_double(f, g, x0, x1, y0, y1, n):
    """
    Monte Carlo integration of f over a domain g>=0, embedded
    in a rectangle [x0,x1]x[y0,y1]. n^2 is the number of
    random points.
    """
    # Draw n**2 random points in the rectangle
    x = np.random.uniform(x0, x1, n)
    y = np.random.uniform(y0, y1, n)
    # Compute sum of f values inside the integration domain
    f_mean = 0
    num_inside = 0   # number of x,y points inside domain (g>=0)
    for i in range(len(x)):
        for j in range(len(y)):
            if g(x[i], y[j]) >= 0:
                num_inside += 1
                f_mean += f(x[i], y[j])
    f_mean = f_mean/float(num_inside)
    area = num_inside/float(n**2)*(x1 - x0)*(y1 - y0)
    return area*f_mean

def test_MonteCarlo_double_rectangle_area():
    """Check the area of a rectangle."""
    def g(x, y):
        return (1 if (0 <= x <= 2 and 3 <= y <= 4.5) else -1)

    x0 = 0;  x1 = 3;  y0 = 2;  y1 = 5  # embedded rectangle
    n = 1000
    np.random.seed(8)      # must fix the seed!
    I_expected = 3.121092  # computed with this seed
    I_computed = MonteCarlo_double(
        lambda x, y: 1, g, x0, x1, y0, y1, n)
    assert abs(I_expected - I_computed) < 1E-14

def test_MonteCarlo_double_circle_r():
    """Check the integral of r over a circle with radius 2."""
    def g(x, y):
        xc, yc = 0, 0  # center
        R = 2          # radius
        return  R**2 - ((x-xc)**2 + (y-yc)**2)

    # Exact: integral of r*r*dr over circle with radius R becomes
    # 2*pi*1/3*R**3
    import sympy
    r = sympy.symbols('r')
    I_exact = sympy.integrate(2*sympy.pi*r*r, (r, 0, 2))
    print 'Exact integral:', I_exact.evalf()
    x0 = -2;  x1 = 2;  y0 = -2;  y1 = 2
    n = 1000
    np.random.seed(6)
    I_expected = 16.7970837117376384  # Computed with this seed
    I_computed = MonteCarlo_double(
        lambda x, y: np.sqrt(x**2 + y**2),
        g, x0, x1, y0, y1, n)
    print 'MC approximation %d samples): %.16f' % (n**2, I_computed)
    assert abs(I_expected - I_computed) < 1E-15

if __name__ == '__main__':
    test_MonteCarlo_double_rectangle_area()
    test_MonteCarlo_double_circle_r()

```


```python
from MC_double import MonteCarlo_double
def g(x, y):
    return (1 if (0 <= x <= 2 and 3 <= y <= 4.5) else -1)
```


```python
MonteCarlo_double(lambda x, y: 1, g, 0, 3, 2, 5, 100)
```




    3.0294




```python
MonteCarlo_double(lambda x, y: 1, g, 0, 3, 2, 5, 1000)
```




    3.0723840000000004




```python
MonteCarlo_double(lambda x, y: 1, g, 0, 3, 2, 5, 1000)
```




    3.0458879999999997




```python
MonteCarlo_double(lambda x, y: 1, g, 0, 3, 2, 5, 2000)
```




    3.0836159999999997




```python
MonteCarlo_double(lambda x, y: 1, g, 0, 3, 2, 5, 2000)
```




    2.8358055




```python
MonteCarlo_double(lambda x, y: 1, g, 0, 3, 2, 5, 5000)
```




    2.89208736



## 3.8 Exercises

* Exercise 3.1: Hand calculations for the trapezoidal method
* Exercise 3.2: Hand calculations for the midpoint method
* Exercise 3.3: Compute a simple integral
* Exercise 3.4: Hand-calculations with sine integrals
* Exercise 3.5: Make test functions for the midpoint method
* Exercise 3.6: Explore rounding errors with large numbers
* Exercise 3.7: Write test functions for
* Exercise 3.8: Rectangle methods
* Exercise 3.9: Adaptive integration
* Exercise 3.10: Integrating x raised to x
* Exercise 3.11: Integrate products of  sinefunctions
* Exercise 3.12: Revisit fit of sines to a function
* Exercise 3.13: Derive the trapezoidal rule for a double integral
* Exercise 3.14: Compute the area of a triangle by Monte Carlo integration
