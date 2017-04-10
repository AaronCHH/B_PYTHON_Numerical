
# coding: utf-8

# # Chapter 6: Optimization

# Robert Johansson
# 
# Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).
# 
# The source code listings can be downloaded from http://www.apress.com/9781484205549

# In[1]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[2]:

import numpy as np


# In[3]:

import sympy


# In[4]:

sympy.init_printing()


# In[5]:

from scipy import optimize


# In[6]:

import cvxopt


# In[7]:

from __future__ import division


# ## Univariate

# In[8]:

r, h = sympy.symbols("r, h")


# In[9]:

Area = 2 * sympy.pi * r**2 + 2 * sympy.pi * r * h


# In[10]:

Volume = sympy.pi * r**2 * h


# In[11]:

h_r = sympy.solve(Volume - 1)[0]


# In[12]:

Area_r = Area.subs(h_r)


# In[13]:

rsol = sympy.solve(Area_r.diff(r))[0]


# In[14]:

rsol


# In[15]:

_.evalf()


# In[16]:

# verify that the second derivative is positive, so that rsol is a minimum
Area_r.diff(r, 2).subs(r, rsol)


# In[17]:

Area_r.subs(r, rsol)


# In[18]:

_.evalf()


# In[19]:

def f(r):
    return 2 * np.pi * r**2 + 2 / r


# In[20]:

r_min = optimize.brent(f, brack=(0.1, 4))


# In[21]:

r_min


# In[22]:

f(r_min)


# In[23]:

optimize.minimize_scalar(f, bracket=(0.1, 5))


# In[24]:

r = np.linspace(0, 2, 100)


# In[25]:

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(r, f(r), lw=2, color='b')
ax.plot(r_min, f(r_min), 'r*', markersize=15)
ax.set_title(r"$f(r) = 2\pi r^2+2/r$", fontsize=18)
ax.set_xlabel(r"$r$", fontsize=18)
ax.set_xticks([0, 0.5, 1, 1.5, 2])
ax.set_ylim(0, 30)

fig.tight_layout()
fig.savefig('ch6-univariate-optimization-example.pdf')


# ## Two-dimensional

# In[26]:

x1, x2 = sympy.symbols("x_1, x_2")


# In[27]:

f_sym = (x1-1)**4 + 5 * (x2-1)**2 - 2*x1*x2


# In[28]:

fprime_sym = [f_sym.diff(x_) for x_ in (x1, x2)]


# In[29]:

# Gradient
sympy.Matrix(fprime_sym)


# In[30]:

fhess_sym = [[f_sym.diff(x1_, x2_) for x1_ in (x1, x2)] for x2_ in (x1, x2)]


# In[31]:

# Hessian
sympy.Matrix(fhess_sym)


# In[32]:

f_lmbda = sympy.lambdify((x1, x2), f_sym, 'numpy')


# In[33]:

fprime_lmbda = sympy.lambdify((x1, x2), fprime_sym, 'numpy')


# In[34]:

fhess_lmbda = sympy.lambdify((x1, x2), fhess_sym, 'numpy')


# In[35]:

def func_XY_X_Y(f):
    """
    Wrapper for f(X) -> f(X[0], X[1])
    """
    return lambda X: np.array(f(X[0], X[1]))


# In[36]:

f = func_XY_X_Y(f_lmbda)


# In[37]:

fprime = func_XY_X_Y(fprime_lmbda)


# In[38]:

fhess = func_XY_X_Y(fhess_lmbda)


# In[39]:

X_opt = optimize.fmin_ncg(f, (0, 0), fprime=fprime, fhess=fhess)


# In[40]:

X_opt


# In[41]:

fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-1, 4, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, f_lmbda(X, Y), 50)
ax.plot(X_opt[0], X_opt[1], 'r*', markersize=15)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)
fig.tight_layout()
fig.savefig('ch6-examaple-two-dim.pdf');


# ## Brute force search for initial point

# In[42]:

def f(X):
    x, y = X
    return (4 * np.sin(np.pi * x) + 6 * np.sin(np.pi * y)) + (x - 1)**2 + (y - 1)**2


# In[43]:

x_start = optimize.brute(f, (slice(-3, 5, 0.5), slice(-3, 5, 0.5)), finish=None)


# In[44]:

x_start


# In[45]:

f(x_start)


# In[46]:

x_opt = optimize.fmin_bfgs(f, x_start)


# In[47]:

x_opt


# In[48]:

f(x_opt)


# In[49]:

def func_X_Y_to_XY(f, X, Y):
    s = np.shape(X)
    return f(np.vstack([X.ravel(), Y.ravel()])).reshape(*s)


# In[50]:

fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-3, 5, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 25)
ax.plot(x_opt[0], x_opt[1], 'r*', markersize=15)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)
fig.tight_layout()
fig.savefig('ch6-example-2d-many-minima.pdf');


# ## Nonlinear least square

# In[51]:

def f(x, beta0, beta1, beta2):
    return beta0 + beta1 * np.exp(-beta2 * x**2)


# In[52]:

beta = (0.25, 0.75, 0.5)


# In[53]:

xdata = np.linspace(0, 5, 50)


# In[54]:

y = f(xdata, *beta)


# In[55]:

ydata = y + 0.05 * np.random.randn(len(xdata))


# In[56]:

def g(beta):
    return ydata - f(xdata, *beta)


# In[57]:

beta_start = (1, 1, 1)


# In[58]:

beta_opt, beta_cov = optimize.leastsq(g, beta_start)


# In[59]:

beta_opt


# In[60]:

fig, ax = plt.subplots()

ax.scatter(xdata, ydata)
ax.plot(xdata, y, 'r', lw=2)
ax.plot(xdata, f(xdata, *beta_opt), 'b', lw=2)
ax.set_xlim(0, 5)
ax.set_xlabel(r"$x$", fontsize=18)
ax.set_ylabel(r"$f(x, \beta)$", fontsize=18)

fig.tight_layout()
fig.savefig('ch6-nonlinear-least-square.pdf')


# In[61]:

beta_opt, beta_cov = optimize.curve_fit(f, xdata, ydata)


# In[62]:

beta_opt


# ## Constrained optimization

# ### Bounds

# In[63]:

def f(X):
    x, y = X
    return (x-1)**2 + (y-1)**2


# In[64]:

x_opt = optimize.minimize(f, (0, 0), method='BFGS').x


# In[65]:

bnd_x1, bnd_x2 = (2, 3), (0, 2)


# In[66]:

x_cons_opt = optimize.minimize(f, np.array([0, 0]), method='L-BFGS-B', bounds=[bnd_x1, bnd_x2]).x


# In[67]:

fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 50)
ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15)
ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15)
bound_rect = plt.Rectangle((bnd_x1[0], bnd_x2[0]), 
                           bnd_x1[1] - bnd_x1[0], bnd_x2[1] - bnd_x2[0],
                           facecolor="grey")
ax.add_patch(bound_rect)
ax.set_xlabel(r"$x_1$", fontsize=18)
ax.set_ylabel(r"$x_2$", fontsize=18)
plt.colorbar(c, ax=ax)

fig.tight_layout()
fig.savefig('ch6-example-constraint-bound.pdf');


# ## Lagrange multiplier

# In[68]:

x = x1, x2, x3, l = sympy.symbols("x_1, x_2, x_3, lambda")


# In[69]:

f = x1 * x2 * x3


# In[70]:

g = 2 * (x1 * x2 + x2 * x3 + x3 * x1) - 1


# In[71]:

L = f + l * g


# In[72]:

grad_L = [sympy.diff(L, x_) for x_ in x]


# In[73]:

sols = sympy.solve(grad_L)
sols


# In[74]:

g.subs(sols[0])


# In[75]:

f.subs(sols[0])


# In[76]:

def f(X):
    return -X[0] * X[1] * X[2]


# In[77]:

def g(X):
    return 2 * (X[0]*X[1] + X[1] * X[2] + X[2] * X[0]) - 1


# In[78]:

constraints = [dict(type='eq', fun=g)]


# In[79]:

result = optimize.minimize(f, [0.5, 1, 1.5], method='SLSQP', constraints=constraints)


# In[80]:

result


# In[81]:

result.x


# ## Inequality constraints

# In[82]:

def f(X):
    return (X[0] - 1)**2 + (X[1] - 1)**2

def g(X):
    return X[1] - 1.75 - (X[0] - 0.75)**4


# In[83]:

x_opt = optimize.minimize(f, (0, 0), method='BFGS').x


# In[84]:

constraints = [dict(type='ineq', fun=g)]


# In[85]:

x_cons_opt = optimize.minimize(f, (0, 0), method='SLSQP', constraints=constraints).x


# In[86]:

x_cons_opt = optimize.minimize(f, (0, 0), method='COBYLA', constraints=constraints).x


# In[87]:

fig, ax = plt.subplots(figsize=(6, 4))
x_ = y_ = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_, y_)
c = ax.contour(X, Y, func_X_Y_to_XY(f, X, Y), 50)
ax.plot(x_opt[0], x_opt[1], 'b*', markersize=15)

ax.plot(x_, 1.75 + (x_-0.75)**4, 'k-', markersize=15)
ax.fill_between(x_, 1.75 + (x_-0.75)**4, 3, color="grey")
ax.plot(x_cons_opt[0], x_cons_opt[1], 'r*', markersize=15)

ax.set_ylim(-1, 3)
ax.set_xlabel(r"$x_0$", fontsize=18)
ax.set_ylabel(r"$x_1$", fontsize=18)
plt.colorbar(c, ax=ax)

fig.tight_layout()
fig.savefig('ch6-example-constraint-inequality.pdf');


# ## Linear programming

# In[88]:

c = np.array([-1.0, 2.0, -3.0])

A = np.array([[ 1.0, 1.0, 0.0],
              [-1.0, 3.0, 0.0],
              [ 0.0, -1.0, 1.0]])

b = np.array([1.0, 2.0, 3.0])


# In[89]:

A_ = cvxopt.matrix(A)
b_ = cvxopt.matrix(b)
c_ = cvxopt.matrix(c)


# In[90]:

sol = cvxopt.solvers.lp(c_, A_, b_)


# In[91]:

x = np.array(sol['x'])


# In[92]:

x


# In[93]:

sol


# In[94]:

sol['primal objective']


# ## Versions

# In[95]:

get_ipython().magic(u'reload_ext version_information')


# In[96]:

get_ipython().magic(u'version_information numpy, scipy, cvxopt, sympy, matplotlib')

