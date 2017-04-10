
# coding: utf-8

# # Chapter 7: Interpolation

# Robert Johansson
# 
# Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).
# 
# The source code listings can be downloaded from http://www.apress.com/9781484205549

# In[1]:

get_ipython().magic(u'matplotlib inline')

import matplotlib as mpl
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = "12"


# In[2]:

import numpy as np


# In[3]:

from numpy import polynomial as P


# In[4]:

from scipy import interpolate


# In[5]:

import matplotlib.pyplot as plt


# In[6]:

from scipy import linalg


# # Polynomials

# In[7]:

p1 = P.Polynomial([1,2,3])


# In[8]:

p1


# In[9]:

p2 = P.Polynomial.fromroots([-1, 1])


# In[10]:

p2


# In[11]:

p1.roots()


# In[12]:

p2.roots()


# In[13]:

p1.coef


# In[14]:

p1.domain


# In[15]:

p1.window


# In[16]:

p1([1.5, 2.5, 3.5])


# In[17]:

p1+p2


# In[18]:

p2 / 5


# In[19]:

p1 = P.Polynomial.fromroots([1, 2, 3])


# In[20]:

p1


# In[21]:

p2 = P.Polynomial.fromroots([2])


# In[22]:

p2


# In[23]:

p3 = p1 // p2


# In[24]:

p3


# In[25]:

p3.roots()


# In[26]:

p2


# In[27]:

c1 = P.Chebyshev([1, 2, 3])


# In[28]:

c1


# In[29]:

c1.roots()


# In[30]:

c = P.Chebyshev.fromroots([-1, 1])


# In[31]:

c


# In[32]:

l = P.Legendre.fromroots([-1, 1])


# In[33]:

l


# In[34]:

c([0.5, 1.5, 2.5])


# In[35]:

l([0.5, 1.5, 2.5])


# # Polynomial interpolation

# In[36]:

x = np.array([1, 2, 3, 4])
y = np.array([1, 3, 5, 4])


# In[37]:

deg = len(x) - 1


# In[38]:

A = P.polynomial.polyvander(x, deg)


# In[39]:

c = linalg.solve(A, y)


# In[40]:

c


# In[41]:

f1 = P.Polynomial(c)


# In[42]:

f1(2.5)


# In[43]:

A = P.chebyshev.chebvander(x, deg)


# In[44]:

c = linalg.solve(A, y)


# In[45]:

c


# In[46]:

f2 = P.Chebyshev(c)


# In[47]:

f2(2.5)


# In[48]:

xx = np.linspace(x.min(), x.max(), 100)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

ax.plot(xx, f1(xx), 'b', lw=2, label='Power basis interp.')
ax.plot(xx, f2(xx), 'r--', lw=2, label='Chebyshev basis interp.')
ax.scatter(x, y, label='data points')

ax.legend(loc=4)
ax.set_xticks(x)
ax.set_ylabel(r"$y$", fontsize=18)
ax.set_xlabel(r"$x$", fontsize=18)

fig.tight_layout()
fig.savefig('ch7-polynomial-interpolation.pdf');


# In[49]:

f1b = P.Polynomial.fit(x, y, deg)


# In[50]:

f1b


# In[51]:

f2b = P.Chebyshev.fit(x, y, deg)


# In[52]:

f2b


# In[53]:

np.linalg.cond(P.chebyshev.chebvander(x, deg))


# In[54]:

np.linalg.cond(P.chebyshev.chebvander((2*x-5)/3.0, deg))


# In[55]:

(2 * x - 5)/3.0


# In[56]:

f1 = P.Polynomial.fit(x, y, 1)
f2 = P.Polynomial.fit(x, y, 2)
f3 = P.Polynomial.fit(x, y, 3)


# In[57]:

xx = np.linspace(x.min(), x.max(), 100)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

ax.plot(xx, f1(xx), 'r', lw=2, label='1st order')
ax.plot(xx, f2(xx), 'g', lw=2, label='2nd order')
ax.plot(xx, f3(xx), 'b', lw=2, label='3rd order')
ax.scatter(x, y, label='data points')

ax.legend(loc=4)
ax.set_xticks(x)
ax.set_ylabel(r"$y$", fontsize=18)
ax.set_xlabel(r"$x$", fontsize=18);


# ## Runge problem

# In[58]:

def runge(x):
    return 1/(1 + 25 * x**2)


# In[59]:

def runge_interpolate(n):
    x = np.linspace(-1, 1, n+1)
    p = P.Polynomial.fit(x, runge(x), deg=n)
    return x, p


# In[60]:

xx = np.linspace(-1, 1, 250)


# In[61]:

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

ax.plot(xx, runge(xx), 'k', lw=2, label="Runge's function")

n = 13
x, p = runge_interpolate(n)
ax.plot(x, runge(x), 'ro')
ax.plot(xx, p(xx), 'r', label='interp. order %d' % n)

n = 14
x, p = runge_interpolate(n)
ax.plot(x, runge(x), 'go')
ax.plot(xx, p(xx), 'g', label='interp. order %d' % n)

ax.legend(loc=8)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1, 2)
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_ylabel(r"$y$", fontsize=18)
ax.set_xlabel(r"$x$", fontsize=18)

fig.tight_layout()
fig.savefig('ch7-polynomial-interpolation-runge.pdf');


# # Spline interpolation

# In[62]:

x = np.linspace(-1, 1, 11)


# In[63]:

y = runge(x)


# In[64]:

f = interpolate.interp1d(x, y, kind=3)


# In[65]:

xx = np.linspace(-1, 1, 100)


# In[66]:

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(xx, runge(xx), 'k', lw=1, label="Runge's function")
ax.plot(x, y, 'ro', label='sample points')
ax.plot(xx, f(xx), 'r--', lw=2, label='spline order 3')

ax.legend()
ax.set_ylim(0, 1.1)
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_ylabel(r"$y$", fontsize=18)
ax.set_xlabel(r"$x$", fontsize=18)

fig.tight_layout()
fig.savefig('ch7-spline-interpolation-runge.pdf');


# In[67]:

x = np.array([0, 1, 2, 3, 4, 5, 6, 7])


# In[68]:

y = np.array([3, 4, 3.5, 2, 1, 1.5, 1.25, 0.9])


# In[69]:

xx = np.linspace(x.min(), x.max(), 100)


# In[70]:

fig, ax = plt.subplots(figsize=(8, 4))

ax.scatter(x, y)

for n in [1, 2, 3, 6]:
    f = interpolate.interp1d(x, y, kind=n)
    ax.plot(xx, f(xx), label='order %d' % n)

ax.legend()
ax.set_ylabel(r"$y$", fontsize=18)
ax.set_xlabel(r"$x$", fontsize=18)

fig.tight_layout()
fig.savefig('ch7-spline-interpolation-orders.pdf');


# # Multivariate interpolation

# ### Regular grid

# In[71]:

x = y = np.linspace(-2, 2, 10)


# In[72]:

def f(x, y):
    return np.exp(-(x + .5)**2 - 2*(y + .5)**2) - np.exp(-(x - .5)**2 - 2*(y - .5)**2)


# In[73]:

X, Y = np.meshgrid(x, y)


# In[74]:

# simulate noisy data at fixed grid points X, Y
Z = f(X, Y) + 0.05 * np.random.randn(*X.shape)


# In[75]:

f_interp = interpolate.interp2d(x, y, Z, kind='cubic')


# In[76]:

xx = yy = np.linspace(x.min(), x.max(), 100)


# In[77]:

ZZi = f_interp(xx, yy)


# In[78]:

XX, YY = np.meshgrid(xx, yy)


# In[79]:

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

c = axes[0].contourf(XX, YY, f(XX, YY), 15, cmap=plt.cm.RdBu)
axes[0].set_xlabel(r"$x$", fontsize=20)
axes[0].set_ylabel(r"$y$", fontsize=20)
axes[0].set_title("exact / high sampling")
cb = fig.colorbar(c, ax=axes[0])
cb.set_label(r"$z$", fontsize=20)

c = axes[1].contourf(XX, YY, ZZi, 15, cmap=plt.cm.RdBu)
axes[1].set_ylim(-2.1, 2.1)
axes[1].set_xlim(-2.1, 2.1)
axes[1].set_xlabel(r"$x$", fontsize=20)
axes[1].set_ylabel(r"$y$", fontsize=20)
axes[1].scatter(X, Y, marker='x', color='k')
axes[1].set_title("interpolation of noisy data / low sampling")
cb = fig.colorbar(c, ax=axes[1])
cb.set_label(r"$z$", fontsize=20)

fig.tight_layout()
fig.savefig('ch7-multivariate-interpolation-regular-grid.pdf')


# In[80]:

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

c = ax.contourf(XX, YY, ZZi, 15, cmap=plt.cm.RdBu)
ax.set_ylim(-2.1, 2.1)
ax.set_xlim(-2.1, 2.1)
ax.set_xlabel(r"$x$", fontsize=20)
ax.set_ylabel(r"$y$", fontsize=20)
ax.scatter(X, Y, marker='x', color='k')
cb = fig.colorbar(c, ax=ax)
cb.set_label(r"$z$", fontsize=20)

fig.tight_layout()
#fig.savefig('ch7-multivariate-interpolation-regular-grid.pdf')


# ### Irregular grid

# In[81]:

np.random.seed(115925231)


# In[82]:

x = y = np.linspace(-1, 1, 100)


# In[83]:

X, Y = np.meshgrid(x, y)


# In[84]:

def f(x, y):
    return np.exp(-x**2 - y**2) * np.cos(4*x) * np.sin(6*y)


# In[85]:

Z = f(X, Y)


# In[86]:

N = 500


# In[87]:

xdata = np.random.uniform(-1, 1, N)


# In[88]:

ydata = np.random.uniform(-1, 1, N)


# In[89]:

zdata = f(xdata, ydata)


# In[90]:

fig, ax = plt.subplots(figsize=(8, 6))
c = ax.contourf(X, Y, Z, 15, cmap=plt.cm.RdBu);
ax.scatter(xdata, ydata, marker='.')
ax.set_ylim(-1,1)
ax.set_xlim(-1,1)
ax.set_xlabel(r"$x$", fontsize=20)
ax.set_ylabel(r"$y$", fontsize=20)

cb = fig.colorbar(c, ax=ax)
cb.set_label(r"$z$", fontsize=20)

fig.tight_layout()
fig.savefig('ch7-multivariate-interpolation-exact.pdf');


# In[91]:

def z_interpolate(xdata, ydata, zdata):
    Zi_0 = interpolate.griddata((xdata, ydata), zdata, (X, Y), method='nearest')
    Zi_1 = interpolate.griddata((xdata, ydata), zdata, (X, Y), method='linear')
    Zi_3 = interpolate.griddata((xdata, ydata), zdata, (X, Y), method='cubic')
    return Zi_0, Zi_1, Zi_3


# In[92]:

fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex=True, sharey=True)

n_vec = [50, 150, 500]

for idx, n in enumerate(n_vec):
    Zi_0, Zi_1, Zi_3 = z_interpolate(xdata[:n], ydata[:n], zdata[:n])
    axes[idx, 0].contourf(X, Y, Zi_0, 15, cmap=plt.cm.RdBu)
    axes[idx, 0].set_ylabel("%d data points\ny" % n, fontsize=16)
    axes[idx, 0].set_title("nearest", fontsize=16)
    axes[idx, 1].contourf(X, Y, Zi_1, 15, cmap=plt.cm.RdBu)
    axes[idx, 1].set_title("linear", fontsize=16)
    axes[idx, 2].contourf(X, Y, Zi_3, 15, cmap=plt.cm.RdBu)
    axes[idx, 2].set_title("cubic", fontsize=16)

for m in range(len(n_vec)):
    axes[idx, m].set_xlabel("x", fontsize=16)
    
fig.tight_layout()
fig.savefig('ch7-multivariate-interpolation-interp.pdf');


# # Versions

# In[93]:

get_ipython().magic(u'reload_ext version_information')


# In[94]:

get_ipython().magic(u'version_information scipy, numpy, matplotlib')

