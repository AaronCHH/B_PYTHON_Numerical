
# 5. Solving Partial Differential Equations
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [5. Solving Partial Differential Equations](#5-solving-partial-differential-equations)
  * [5.1 Finite Difference Methods](#51-finite-difference-methods)
    * [5.1.1 Reduction of a PDE to a System of ODEs](#511-reduction-of-a-pde-to-a-system-of-odes)
    * [5.1.2 Construction of a Test Problem with Known Discrete Solution](#512-construction-of-a-test-problem-with-known-discrete-solution)
    * [5.1.3 Implementation: Forward Euler Method](#513-implementation-forward-euler-method)
    * [5.1.4 Application: Heat Conduction in a Rod](#514-application-heat-conduction-in-a-rod)
    * [5.1.5 Vectorization](#515-vectorization)
    * [5.1.6 Using Odespy to Solve the System of ODEs](#516-using-odespy-to-solve-the-system-of-odes)
    * [5.1.7 Implicit Methods](#517-implicit-methods)
  * [5.2 Exercises](#52-exercises)

<!-- tocstop -->


## 5.1 Finite Difference Methods

### 5.1.1 Reduction of a PDE to a System of ODEs

### 5.1.2 Construction of a Test Problem with Known Discrete Solution

### 5.1.3 Implementation: Forward Euler Method


```python
# %load py/test_diffusion_pde_exact_linear.py
"""Verify the implementation of the diffusion equation."""

from ode_system_FE import ode_FE
from numpy import linspace, zeros, abs

def rhs(u, t):
    N = len(u) - 1
    rhs = zeros(N+1)
    rhs[0] = dsdt(t)
    for i in range(1, N):
        rhs[i] = (beta/dx**2)*(u[i+1] - 2*u[i] + u[i-1]) + \
                 f(x[i], t)
    rhs[N] = (beta/dx**2)*(2*u[N-1] + 2*dx*dudx(t) -
                           2*u[N]) + f(x[N], t)
    return rhs

def u_exact(x, t):
    return (3*t + 2)*(x - L)

def dudx(t):
    return (3*t + 2)

def s(t):
    return u_exact(0, t)

def dsdt(t):
    return 3*(-L)

def f(x, t):
    return 3*(x-L)


def verify_sympy_ForwardEuler():
    import sympy as sp
    beta, x, t, dx, dt, L = sp.symbols('beta x t dx dt L')
    u = lambda x, t: (3*t + 2)*(x - L)**2
    f = lambda x, t, beta, L: 3*(x-L)**2 - (3*t + 2)*2*beta
    s = lambda t: (3*t + 2)*L**2
    N = 4
    rhs = [None]*(N+1)
    rhs[0] = sp.diff(s(t), t)
    for i in range(1, N):
        rhs[i] = (beta/dx**2)*(u(x+dx,t) - 2*u(x,t) + u(x-dx,t)) + \
                 f(x, t, beta, L)
    rhs[N] = (beta/dx**2)*(u(x-dx,t) + 2*dx*(3*t+2) -
                           2*u(x,t) + u(x-dx,t)) + f(x, t, beta, L)
    for i in range(len(rhs)):
        rhs[i] = sp.simplify(sp.expand(rhs[i])).subs(x, i*dx)
        print rhs[i]
        lhs = (u(x, t+dt) - u(x,t))/dt  # Forward Euler difference
        lhs = sp.simplify(sp.expand(lhs.subs(x, i*dx)))
        print lhs
        print sp.simplify(lhs - rhs[i])
        print '---'

def test_diffusion_exact_linear():
    global beta, dx, L, x  # needed in rhs
    L = 1.5
    beta = 0.5
    N = 4
    x = linspace(0, L, N+1)
    dx = x[1] - x[0]
    u = zeros(N+1)

    U_0 = zeros(N+1)
    U_0[0] = s(0)
    U_0[1:] = u_exact(x[1:], 0)
    dt = 0.1
    print dt

    u, t = ode_FE(rhs, U_0, dt, T=1.2)

    tol = 1E-12
    for i in range(0, u.shape[0]):
        diff = abs(u_exact(x, t[i]) - u[i,:]).max()
        assert diff < tol, 'diff=%.16g' % diff
        print 'diff=%g at t=%g' % (diff, t[i])


if __name__ == '__main__':
    test_diffusion_exact_linear()
    verify_sympy_ForwardEuler()

```

### 5.1.4 Application: Heat Conduction in a Rod


```python
# %load py/rod_FE.py
"""Temperature evolution in a rod, computed by a ForwardEuler method."""

from numpy import linspace, zeros, linspace
import time

def rhs(u, t):
    N = len(u) - 1
    rhs = zeros(N+1)
    rhs[0] = dsdt(t)
    for i in range(1, N):
        rhs[i] = (beta/dx**2)*(u[i+1] - 2*u[i] + u[i-1]) + \
                 g(x[i], t)
    i = N
    rhs[i] = (beta/dx**2)*(2*u[i-1] + 2*dx*dudx(t) -
                           2*u[i]) + g(x[N], t)
    return rhs

def dudx(t):
    return 0

def s(t):
    return 323

def dsdt(t):
    return 0

def g(x, t):
    return 0


L = 0.5
beta = 8.2E-5
N = 40
x = linspace(0, L, N+1)
dx = x[1] - x[0]
u = zeros(N+1)

U_0 = zeros(N+1)
U_0[0] = s(0)
U_0[1:] = 283
dt = dx**2/(2*beta)
print 'stability limit:', dt
#dt = 0.00034375

t0 = time.clock()
from ode_system_FE import ode_FE
u, t = ode_FE(rhs, U_0, dt, T=1*60*60)
t1 = time.clock()
print 'CPU time: %.1fs' % (t1 - t0)

# Make movie
import os
os.system('rm tmp_*.png')
import matplotlib.pyplot as plt
plt.ion()
y = u[0,:]
lines = plt.plot(x, y)
plt.axis([x[0], x[-1], 273, s(0)+10])
plt.xlabel('x')
plt.ylabel('u(x,t)')
counter = 0
# Plot each of the first 100 frames, then increase speed by 10x
change_speed = 100
for i in range(0, u.shape[0]):
    print t[i]
    plot = True if i <= change_speed else i % 10 == 0
    lines[0].set_ydata(u[i,:])
    if i > change_speed:
        plt.legend(['t=%.0f 10x' % t[i]])
    else:
        plt.legend(['t=%.0f' % t[i]])
    plt.draw()
    if plot:
        plt.savefig('tmp_%04d.png' % counter)
        counter += 1
    #time.sleep(0.2)

```

### 5.1.5 Vectorization

### 5.1.6 Using Odespy to Solve the System of ODEs

### 5.1.7 Implicit Methods


```python
# %load py/rod_BE.py
"""Temperature evolution in a rod, computed by a BackwardEuler method."""

from numpy import linspace, zeros, linspace

def rhs(u, t):
    N = len(u) - 1
    rhs = zeros(N+1)
    rhs[0] = dsdt(t)
    for i in range(1, N):
        rhs[i] = (beta/dx**2)*(u[i+1] - 2*u[i] + u[i-1]) + \
                 g(x[i], t)
    rhs[N] = (beta/dx**2)*(2*u[i-1] + 2*dx*dudx(t) -
                           2*u[i]) + g(x[N], t)
    return rhs

def K(u, t):
    N = len(u) - 1
    K = zeros((N+1,N+1))
    K[0,0] = 0
    for i in range(1, N):
        K[i,i-1] = beta/dx**2
        K[i,i] = -2*beta/dx**2
        K[i,i+1] = beta/dx**2
    K[N,N-1] = (beta/dx**2)*2
    K[N,N] = (beta/dx**2)*(-2)
    return K

def rhs_vec(u, t):
    N = len(u) - 1
    rhs = zeros(N+1)
    rhs[0] = dsdt(t)
    rhs[1:N] = (beta/dx**2)*(u[2:N+1] - 2*u[1:N] + u[0:N-1]) + \
               g(x[1:N], t)
    i = N
    rhs[i] = (beta/dx**2)*(2*u[i-1] + 2*dx*dudx(t) -
                           2*u[i]) + g(x[N], t)
    return rhs

def K_vec(u, t):
    """Vectorized computation of K."""
    N = len(u) - 1
    K = zeros((N+1,N+1))
    K[0,0] = 0
    K[1:N-1] = beta/dx**2
    K[1:N] = -2*beta/dx**2
    K[2:N+1] = beta/dx**2
    K[N,N-1] = (beta/dx**2)*2
    K[N,N] = (beta/dx**2)*(-2)
    return K

def dudx(t):
    return 0

def s(t):
    return 323

def dsdt(t):
    return 0

def g(x, t):
    return 0

L = 0.5
beta = 8.2E-5
N = 40
x = linspace(0, L, N+1)
dx = x[1] - x[0]
u = zeros(N+1)

U_0 = zeros(N+1)
U_0[0] = s(0)
U_0[1:] = 283
dt = dx**2/(2*beta)
print 'stability limit:', dt
dt = 600  # 10 min

import odespy
solver = odespy.BackwardEuler(rhs, f_is_linear=True, jac=K)
solver = odespy.ThetaRule(rhs, f_is_linear=True, jac=K, theta=0.5)
solver.set_initial_condition(U_0)
T = 1*60*60
N_t = int(round(T/float(dt)))
time_points = linspace(0, T, N_t+1)
u, t = solver.solve(time_points)

# Make movie
import os
os.system('rm tmp_*.png')
import matplotlib.pyplot as plt
import time
plt.ion()
y = u[0,:]
lines = plt.plot(x, y)
plt.axis([x[0], x[-1], 273, s(0)+10])
plt.xlabel('x')
plt.ylabel('u(x,t)')
counter = 0
for i in range(0, u.shape[0]):
    print t[i]
    lines[0].set_ydata(u[i,:])
    plt.legend(['t=%.0f' % t[i]])
    plt.draw()
    plt.savefig('tmp_%04d.png' % counter)
    counter += 1
    time.sleep(0.2)

```

## 5.2 Exercises

* Exercise 5.1: Simulate a diffusion equation by hand
* Exercise 5.2: Compute temperature variations in the round
* Exercise 5.3: Compare implicit methods
* Exercise 5.4: Explore adaptive and implicit methods
* Exercise 5.5: Investigate the theta rule
* Exercise 5.6: Compute the diffusion of a Gaussian peak
* Exercise 5.7: Vectorize a function for computing the area of apolygon
* Exercise 5.8: Explore symmetry
* Exercise 5.9: Compute solutions as t -> infinit
* Exercise 5.10: Solve a two-point boundary value problem



```python

```
