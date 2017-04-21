
# Ch04 Solving Ordinary Differential Equations
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Ch04 Solving Ordinary Differential Equations](#ch04-solving-ordinary-differential-equations)
  * [4.1 Population Growth](#41-population-growth)
    * [4.1.1 Derivation of the Model](#411-derivation-of-the-model)
    * [4.1.2 Numerical Solution](#412-numerical-solution)
    * [4.1.3 Programming the Forward Euler Scheme; the Special Case](#413-programming-the-forward-euler-scheme-the-special-case)
    * [4.1.4 Understanding the Forward Euler Method](#414-understanding-the-forward-euler-method)
    * [4.1.5 Programming the Forward Euler Scheme; the General Case](#415-programming-the-forward-euler-scheme-the-general-case)
    * [4.1.6 Making the Population Growth Model More Realistic](#416-making-the-population-growth-model-more-realistic)
    * [4.1.7 Verification: ExactLinear Solution of theDiscrete Equations](#417-verification-exactlinear-solution-of-thediscrete-equations)
  * [4.2 Spreading of Diseases](#42-spreading-of-diseases)
    * [4.2.1 Spreading of a Flu](#421-spreading-of-a-flu)
    * [4.2.2 A Forward Euler Method for the Differential Equation System](#422-a-forward-euler-method-for-the-differential-equation-system)
    * [4.2.3 Programming the Numerical Method; the Special Case](#423-programming-the-numerical-method-the-special-case)
    * [4.2.4 Outbreak or Not](#424-outbreak-or-not)
    * [4.2.5 Abstract Problem and Notation](#425-abstract-problem-and-notation)
    * [4.2.6 Programming the Numerical Method; the General Case](#426-programming-the-numerical-method-the-general-case)
    * [4.2.7 Time-Restricted Immunity](#427-time-restricted-immunity)
    * [4.2.8 Incorporating Vaccination](#428-incorporating-vaccination)
    * [4.2.9 Discontinuous Coefficients: A Vaccination Campaign](#429-discontinuous-coefficients-a-vaccination-campaign)
  * [4.3 Oscillating One-Dimensional Systems](#43-oscillating-one-dimensional-systems)
    * [4.3.1 Derivation of a Simple Model](#431-derivation-of-a-simple-model)
    * [4.3.2 Numerical Solution](#432-numerical-solution)
    * [4.3.3 Programming the Numerical Method; the Special Case](#433-programming-the-numerical-method-the-special-case)
    * [4.3.4 A Magic Fix of the Numerical Metho](#434-a-magic-fix-of-the-numerical-metho)
    * [4.3.5 The 2nd-Order Runge-Kutta Method (or Heun’s Method)](#435-the-2nd-order-runge-kutta-method-or-heuns-method)
    * [4.3.6 Software for Solving ODEs](#436-software-for-solving-odes)
    * [4.3.7 The 4th-Order Runge-Kutta Method](#437-the-4th-order-runge-kutta-method)
    * [4.3.8 More Effects: Damping, Nonlinearity, and External Forces](#438-more-effects-damping-nonlinearity-and-external-forces)
    * [4.3.9 Illustration of Linear Damping](#439-illustration-of-linear-damping)
    * [4.3.10 Illustration of Linear Damping with Sinusoidal Excitation](#4310-illustration-of-linear-damping-with-sinusoidal-excitation)
    * [4.3.11 Spring-Mass System with Sliding Friction](#4311-spring-mass-system-with-sliding-friction)
    * [4.3.12 A finite Difference Method; Undamped, Linear Case](#4312-a-finite-difference-method-undamped-linear-case)
    * [4.3.13 A Finite Difference Method; Linear Damping](#4313-a-finite-difference-method-linear-damping)
  * [4.4 Exercises](#44-exercises)

<!-- tocstop -->



```python
import sys
sys.path.append("./py")
```

## 4.1 Population Growth

### 4.1.1 Derivation of the Model

### 4.1.2 Numerical Solution

### 4.1.3 Programming the Forward Euler Scheme; the Special Case


```python
# %load py/growth1.py
N_0 = input('Give initial population size N_0: ')
r   = input('Give net growth rate r: ')
dt  = input('Give time step size: ')
N_t = input('Give number of steps: ')
from numpy import linspace, zeros
t = linspace(0, (N_t+1)*dt, N_t+2)
N = zeros(N_t+2)

N[0] = N_0
for n in range(N_t+1):
    N[n+1] = N[n] + r*dt*N[n]

import matplotlib.pyplot as plt
numerical_sol = 'bo' if N_t < 70 else 'b-'
plt.plot(t, N, numerical_sol, t, N_0*exp(r*t), 'r-')
plt.legend(['numerical', 'exact'], loc='upper left')
plt.xlabel('t'); plt.ylabel('N(t)')
filestem = 'growth1_%dsteps' % N_t
plt.savefig('%s.png' % filestem); plt.savefig('%s.pdf' % filestem)

```

### 4.1.4 Understanding the Forward Euler Method

### 4.1.5 Programming the Forward Euler Scheme; the General Case


```python
# %load py/ode_FE.py
from numpy import linspace, zeros, exp
import matplotlib.pyplot as plt

def ode_FE(f, U_0, dt, T):
    N_t = int(round(float(T)/dt))
    u = zeros(N_t+1)
    t = linspace(0, N_t*dt, len(u))
    u[0] = U_0
    for n in range(N_t):
        u[n+1] = u[n] + dt*f(u[n], t[n])
    return u, t

def demo_population_growth():
    """Test case: u'=r*u, u(0)=100."""
    def f(u, t):
        return 0.1*u

    u, t = ode_FE(f=f, U_0=100, dt=0.5, T=20)
    plt.plot(t, u, t, 100*exp(0.1*t))
    plt.show()

if __name__ == '__main__':
    demo_population_growth()

```

### 4.1.6 Making the Population Growth Model More Realistic


```python
# %load py/logistic.py
from ode_FE import ode_FE
import matplotlib.pyplot as plt

for dt, T in zip((0.5, 20), (60, 100)):
    u, t = ode_FE(f=lambda u, t: 0.1*(1 - u/500.)*u, \
                               U_0=100, dt=dt, T=T)
    plt.figure()  # Make separate figures for each pass in the loop
    plt.plot(t, u, 'b-')
    plt.xlabel('t'); plt.ylabel('N(t)')
    plt.savefig('tmp_%g.png' % dt); plt.savefig('tmp_%g.pdf' % dt)

```

### 4.1.7 Verification: ExactLinear Solution of theDiscrete Equations


```python
# %load py/test_ode_FE_exact_linear.py
from ode_FE import ode_FE

def test_ode_FE():
    """Test that a linear u(t)=a*t+b is exactly reproduced."""

    def exact_solution(t):
        return a*t + b

    def f(u, t):  # ODE
        return a + (u - exact_solution(t))**m

    a = 4
    b = -1
    m = 6

    dt = 0.5
    T = 20.0

    u, t = ode_FE(f, exact_solution(0), dt, T)
    diff = abs(exact_solution(t) - u).max()
    tol = 1E-15           # Tolerance for float comparison
    success = diff < tol
    assert success

test_ode_FE()

```

## 4.2 Spreading of Diseases

### 4.2.1 Spreading of a Flu

### 4.2.2 A Forward Euler Method for the Differential Equation System

### 4.2.3 Programming the Numerical Method; the Special Case


```python
# %load py/SIR2.py
"""As the basic SIR1.py, but including loss of immunity."""

from numpy import zeros, linspace
import matplotlib.pyplot as plt

# Time unit: 1 h
beta = 10./(40*8*24)
beta /= 4            # Reduce beta compared to SIR1.py
print 'beta:', beta
gamma = 3./(15*24)
dt = 0.1             # 6 min
D = 300              # Simulate for D days
N_t = int(D*24/dt)   # Corresponding no of hours
nu = 1./(24*90)      # Average loss of immunity: 50 days

t = linspace(0, N_t*dt, N_t+1)
S = zeros(N_t+1)
I = zeros(N_t+1)
R = zeros(N_t+1)

# Initial condition
S[0] = 50
I[0] = 1
R[0] = 0

# Step equations forward in time
for n in range(N_t):
    S[n+1] = S[n] - dt*beta*S[n]*I[n] + dt*nu*R[n]
    I[n+1] = I[n] + dt*beta*S[n]*I[n] - dt*gamma*I[n]
    R[n+1] = R[n] + dt*gamma*I[n] - dt*nu*R[n]

fig = plt.figure()
l1, l2, l3 = plt.plot(t, S, t, I, t, R)
fig.legend((l1, l2, l3), ('S', 'I', 'R'), 'upper left')
plt.xlabel('hours')
plt.show()
plt.savefig('tmp.pdf'); plt.savefig('tmp.png')

```

### 4.2.4 Outbreak or Not

### 4.2.5 Abstract Problem and Notation

### 4.2.6 Programming the Numerical Method; the General Case


```python
# %load py/ode_system_FE.py
from numpy import linspace, zeros, asarray
import matplotlib.pyplot as plt

def ode_FE(f, U_0, dt, T):
    N_t = int(round(float(T)/dt))
    # Ensure that any list/tuple returned from f_ is wrapped as array
    f_ = lambda u, t: asarray(f(u, t))
    u = zeros((N_t+1, len(U_0)))
    t = linspace(0, N_t*dt, len(u))
    u[0] = U_0
    for n in range(N_t):
        u[n+1] = u[n] + dt*f_(u[n], t[n])
    return u, t

def demo_SIR():
    """Test case using a SIR model."""
    def f(u, t):
        S, I, R = u
        return [-beta*S*I, beta*S*I - gamma*I, gamma*I]

    beta = 10./(40*8*24)
    gamma = 3./(15*24)
    dt = 0.1             # 6 min
    D = 30               # Simulate for D days
    N_t = int(D*24/dt)   # Corresponding no of hours
    T = dt*N_t           # End time
    U_0 = [50, 1, 0]

    u, t = ode_FE(f, U_0, dt, T)

    S = u[:,0]
    I = u[:,1]
    R = u[:,2]
    fig = plt.figure()
    l1, l2, l3 = plt.plot(t, S, t, I, t, R)
    fig.legend((l1, l2, l3), ('S', 'I', 'R'), 'lower right')
    plt.xlabel('hours')
    plt.show()

    # Consistency check:
    N = S[0] + I[0] + R[0]
    eps = 1E-12  # Tolerance for comparing real numbers
    for n in range(len(S)):
        success = abs(S[n] + I[n] + R[n] - N) < eps
        assert success

if __name__ == '__main__':
    demo_SIR()

```

### 4.2.7 Time-Restricted Immunity


```python
for n in range(N_t):
    S[n+1] = S[n] - dt*beta*S[n]*I[n] + dt*nu*R[n]
    I[n+1] = I[n] + dt*beta*S[n]*I[n] - dt*gamma*I[n]
    R[n+1] = R[n] + dt*gamma*I[n] - dt*nu*R[n]
```

### 4.2.8 Incorporating Vaccination

### 4.2.9 Discontinuous Coefficients: A Vaccination Campaign


```python
def p(t):
    return 0.005 if (6*24 <= t <= 15*24) else 0
```


```python
p = zeros(N_t+1)
start_index = 6*24/dt
stop_index = 15*24/dt
p[start_index: stop_index] = 0.005
```

## 4.3 Oscillating One-Dimensional Systems

### 4.3.1 Derivation of a Simple Model

### 4.3.2 Numerical Solution

### 4.3.3 Programming the Numerical Method; the Special Case


```python
# %load py/osc_EC.py
from numpy import zeros, linspace, pi, cos
import matplotlib.pyplot as plt

omega = 2
P = 2*pi/omega
dt = P/20
T = 40*P
T = P
N_t = int(round(T/dt))
t = linspace(0, N_t*dt, N_t+1)
print 'N_t:', N_t

u = zeros(N_t+1)
v = zeros(N_t+1)

# Initial condition
X_0 = 2
u[0] = X_0
v[0] = 0

# Step equations forward in time
for n in range(N_t):
    v[n+1] = v[n] - dt*omega**2*u[n]
    u[n+1] = u[n] + dt*v[n+1]

# Plot the last four periods to illustrate the accuracy
# in long time simulations
N4l = int(round(4*P/dt))  # No of intervals to be plotted
fig = plt.figure()
l1, l2 = plt.plot(t[-N4l:], u[-N4l:], 'b-',
                  t[-N4l:], X_0*cos(omega*t)[-N4l:], 'r--')
fig.legend((l1, l2), ('numerical', 'exact'), 'upper left')
plt.xlabel('t')
plt.show()
plt.savefig('tmp.pdf'); plt.savefig('tmp.png')
print '%.16f %.16f' % (u[-1], v[-1])

```

### 4.3.4 A Magic Fix of the Numerical Metho

### 4.3.5 The 2nd-Order Runge-Kutta Method (or Heun’s Method)

### 4.3.6 Software for Solving ODEs


```python
%load py/osc_odespy.py
```

### 4.3.7 The 4th-Order Runge-Kutta Method

### 4.3.8 More Effects: Damping, Nonlinearity, and External Forces


```python
# %load py/osc_EC_general.py
from matplotlib.pyplot import plot, hold, legend, \
     xlabel, ylabel, savefig, title, figure, show

def EulerCromer(f, s, F, m, T, U_0, V_0, dt):
    from numpy import zeros, linspace
    N_t = int(round(T/dt))
    print 'N_t:', N_t
    t = linspace(0, N_t*dt, N_t+1)

    u = zeros(N_t+1)
    v = zeros(N_t+1)

    # Initial condition
    u[0] = U_0
    v[0] = V_0

    # Step equations forward in time
    for n in range(N_t):
        v[n+1] = v[n] + dt*(1./m)*(F(t[n]) - f(v[n]) - s(u[n]))
        u[n+1] = u[n] + dt*v[n+1]
    return u, v, t

def test_undamped_linear():
    """Compare with data from osc_EC.py in a linear problem."""
    from numpy import pi
    omega = 2
    P = 2*pi/omega
    dt = P/20
    T = 40*P
    exact_v = -3.5035725322034139
    exact_u = 0.7283057044967003
    computed_u, computed_v, t = EulerCromer(
        f=lambda v: 0, s=lambda u: omega**2*u,
        F=lambda t: 0, m=1, T=T, U_0=2, V_0=0, dt=dt)
    diff_u = abs(exact_u - computed_u[-1])
    diff_v = abs(exact_v - computed_v[-1])
    tol = 1E-14
    assert diff_u < tol and diff_v < tol

def _test_manufactured_solution(damping=True):
    import sympy as sp
    t, m, k, b = sp.symbols('t m k b')
    # Choose solution
    u = sp.sin(t)
    v = sp.diff(u, t)
    # Choose f, s, F
    f = b*v
    s = k*sp.tanh(u)
    F = sp.cos(2*t)

    equation = m*sp.diff(v, t) + f + s - F

    # Adjust F (source term because of manufactured solution)
    F += equation
    print 'F:', F

    # Set values for the symbols m, b, k
    m = 0.5
    k = 1.5
    b = 0.5 if damping else 0
    F = F.subs('m', m).subs('b', b).subs('k', k)

    print f, s, F
    # Turn sympy expression into Python function
    F = sp.lambdify([t], F)
    # Define Python functions for f and s
    # (the expressions above are functions of t, we need
    # s(u) and f(v)
    from numpy import tanh
    s = lambda u: k*tanh(u)
    f = lambda v: b*v

    # Add modules='numpy' such that exact u and v work
    # with t as array argument
    exact_u = sp.lambdify([t], u, modules='numpy')
    exact_v = sp.lambdify([t], v, modules='numpy')


    # Solve problem for different dt
    from numpy import pi, sqrt, sum, log
    P = 2*pi
    time_intervals_per_period = [20, 40, 80, 160, 240]
    h   = []  # store discretization parameters
    E_u = []  # store errors in u
    E_v = []  # store errors in v

    for n in time_intervals_per_period:
        dt = P/n
        T = 8*P
        computed_u, computed_v, t = EulerCromer(
            f=f, s=s, F=F, m=m, T=T,
            U_0=exact_u(0), V_0=exact_v(0), dt=dt)

        error_u = sqrt(dt*sum((exact_u(t) - computed_u)**2))
        error_v = sqrt(dt*sum((exact_v(t) - computed_v)**2))
        h.append(dt)
        E_u.append(error_u)
        E_v.append(error_v)

        """
        # Compare exact and computed curves for this resolution
        figure()
        plot_u(computed_u, t, show=False)
        hold('on')
        plot(t, exact_u(t), show=True)
        legend(['numerical', 'exact'])
        savefig('tmp_%d.pdf' % n); savefig('tmp_%d.png' % n)
        """
    # Compute convergence rates
    r_u = [log(E_u[i]/E_u[i-1])/log(h[i]/h[i-1])
           for i in range(1, len(h))]
    r_v = [log(E_u[i]/E_u[i-1])/log(h[i]/h[i-1])
           for i in range(1, len(h))]
    tol = 0.02
    exact_r_u = 1.0 if damping else 2.0
    exact_r_v = 1.0 if damping else 2.0
    success = abs(exact_r_u - r_u[-1]) < tol and \
              abs(exact_r_v - r_v[-1]) < tol
    msg = ' u rate: %.2f, v rate: %.2f' % (r_u[-1], r_v[-1])
    assert success, msg

def test_manufactured_solution():
    _test_manufactured_solution(damping=True)
    _test_manufactured_solution(damping=False)

# Plot the a percentage of the time series, up to the end, to
# illustrate the accuracy in long time simulations
def plot_u(u, t, percentage=100, show=True, heading='', labels=('t', 'u')):
    index = len(u)*percentage/100.
    plot(t[-index:], u[-index:], 'b-', show=show)
    xlabel(labels[0]);  ylabel(labels[1])
    title(heading)
    savefig('tmp.pdf'); savefig('tmp.png')
    if show:
        show()

def linear_damping():
    b = 0.3
    f = lambda v: b*v
    s = lambda u: k*u
    F = lambda t: 0

    m = 1
    k = 1
    U_0 = 1
    V_0 = 0

    T = 12*pi
    dt = T/5000.

    u, v, t = EulerCromer(f=f, s=s, F=F, m=m, T=T,
                          U_0=U_0, V_0=V_0, dt=dt)
    plot_u(u, t)

def linear_damping_sine_excitation():
    b = 0.3
    f = lambda v: b*v
    s = lambda u: k*u
    from math import pi, sin
    w = 1
    A = 0.5
    F = lambda t: A*sin(w*t)

    m = 1
    k = 1
    U_0 = 1
    V_0 = 0

    T = 12*pi
    dt = T/5000.

    u, v, t = EulerCromer(f=f, s=s, F=F, m=m, T=T,
                          U_0=U_0, V_0=V_0, dt=dt)
    plot_u(u, t)

def sliding_friction():
    from numpy import tanh, sign

    f = lambda v: mu*m*g*sign(v)
    alpha = 60.0
    s = lambda u: k/alpha*tanh(alpha*u)
    s = lambda u: k*u
    F = lambda t: 0

    g = 9.81
    mu = 0.4
    m = 1
    k = 1000

    U_0 = 0.1
    V_0 = 0

    T = 2
    dt = T/5000.

    u, v, t = EulerCromer(f=f, s=s, F=F, m=m, T=T,
                          U_0=U_0, V_0=V_0, dt=dt)
    plot_u(u, t)

if __name__ == '__main__':
    test_undamped_linear()
    test_manufactured_solution()
    #sliding_friction()
    linear_damping_sine_excitation()

```

### 4.3.9 Illustration of Linear Damping


```python
def linear_damping():
    b = 0.3
    f = lambda v: b*v
    s = lambda u: k*u
    F = lambda t: 0
    m = 1
    k = 1
    U_0 = 1
    V_0 = 0
    T = 12*pi
    dt = T/5000.
    u, v, t = EulerCromer(f=f, s=s, F=F, m=m, T=T,
                          U_0=U_0, V_0=V_0, dt=dt)
    plot_u(u, t)
```

### 4.3.10 Illustration of Linear Damping with Sinusoidal Excitation

### 4.3.11 Spring-Mass System with Sliding Friction


```python
def sliding_friction():
    from numpy import tanh, sign
    f = lambda v: mu*m*g*sign(v)
    alpha = 60.0
    s = lambda u: k/alpha*tanh(alpha*u)
    F = lambda t: 0
    g = 9.81
    mu = 0.4
    m = 1
    k = 1000
    U_0 = 0.1
    V_0 = 0
    T = 2
    dt = T/5000.
    u, v, t = EulerCromer(f=f, s=s, F=F, m=m, T=T,
                          U_0=U_0, V_0=V_0, dt=dt)
    plot_u(u, t)
```

### 4.3.12 A finite Difference Method; Undamped, Linear Case


```python
# %load py/osc_2nd_order.py
from numpy import zeros, linspace

def osc_2nd_order(U_0, omega, dt, T):
    """
    Solve u'' + omega**2*u = 0 for t in (0,T], u(0)=U_0 and u'(0)=0,
    by a central finite difference method with time step dt.
    """
    dt = float(dt)
    Nt = int(round(T/dt))
    u = zeros(Nt+1)
    t = linspace(0, Nt*dt, Nt+1)

    u[0] = U_0
    u[1] = u[0] - 0.5*dt**2*omega**2*u[0]
    for n in range(1, Nt):
        u[n+1] = 2*u[n] - u[n-1] - dt**2*omega**2*u[n]
    return u, t

```

### 4.3.13 A Finite Difference Method; Linear Damping

## 4.4 Exercises

* Exercise 4.1: Geometric construction of the Forward Euler method
* Exercise 4.2: Maketest functions for the Forward Euler method
* Exercise 4.3: Implement and evaluate Heun'smethod
* Exercise 4.4: Find an appropriate time step; logistic model
* Exercise 4.5: Find an appropriate time step; SIR model
* Exercise 4.6: Model an adaptiv evaccination campaign
* Exercise 4.7: Makea SIRV model with time-limited effect of vaccination
* Exercise 4.8: Refactor a flat program
* Exercise 4.9: Simulate oscillations by a general ODEsolver
* Exercise 4.10: Compute the energy in oscillations
* Exercise 4.11: Use a Backward Euler scheme for population growth
* Exercise 4.12: Use a Crank-Nicolson scheme for population growth
* Exercise 4.13: Und erstandfinitedifferencesviaTaylorseries
* Exercise 4.14: Use a Backward Euler scheme for oscillations
* Exercise 4.15: Use Heun's method for the SIR model
* Exercise 4.16: Use Odes.py to solve a simpleODE
* Exercise 4.17: Setup a Backward Euler scheme for oscillations
* Exercise 4.18: Setup a Forward Euler scheme for nonlinear and damped
* Exercise 4.19: Discretize an initial condition


```python

```
