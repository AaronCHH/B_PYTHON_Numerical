
# Chapter 7: Interpolation
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Chapter 7: Interpolation](#chapter-7-interpolation)
* [Polynomials](#polynomials)
* [Polynomial interpolation](#polynomial-interpolation)
  * [Runge problem](#runge-problem)
* [Spline interpolation](#spline-interpolation)
* [Multivariate interpolation](#multivariate-interpolation)
    * [Regular grid](#regular-grid)
    * [Irregular grid](#irregular-grid)
* [Versions](#versions)

<!-- tocstop -->


---

Robert Johansson

Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).

The source code listings can be downloaded from http://www.apress.com/9781484205549


```python
%matplotlib inline

import matplotlib as mpl
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = "12"
```


```python
import numpy as np
```


```python
from numpy import polynomial as P
```


```python
from scipy import interpolate
```


```python
import matplotlib.pyplot as plt
```


```python
from scipy import linalg
```

# Polynomials


```python
p1 = P.Polynomial([1,2,3])
```


```python
p1
```




    Polynomial([ 1.,  2.,  3.], [-1,  1], [-1,  1])




```python
p2 = P.Polynomial.fromroots([-1, 1])
```


```python
p2
```




    Polynomial([-1.,  0.,  1.], [-1.,  1.], [-1.,  1.])




```python
p1.roots()
```




    array([-0.33333333-0.47140452j, -0.33333333+0.47140452j])




```python
p2.roots()
```




    array([-1.,  1.])




```python
p1.coef
```




    array([ 1.,  2.,  3.])




```python
p1.domain
```




    array([-1,  1])




```python
p1.window
```




    array([-1,  1])




```python
p1([1.5, 2.5, 3.5])
```




    array([ 10.75,  24.75,  44.75])




```python
p1+p2
```




    Polynomial([ 0.,  2.,  4.], [-1.,  1.], [-1.,  1.])




```python
p2 / 5
```




    Polynomial([-0.2,  0. ,  0.2], [-1.,  1.], [-1.,  1.])




```python
p1 = P.Polynomial.fromroots([1, 2, 3])
```


```python
p1
```




    Polynomial([ -6.,  11.,  -6.,   1.], [-1.,  1.], [-1.,  1.])




```python
p2 = P.Polynomial.fromroots([2])
```


```python
p2
```




    Polynomial([-2.,  1.], [-1.,  1.], [-1.,  1.])




```python
p3 = p1 // p2
```


```python
p3
```




    Polynomial([ 3., -4.,  1.], [-1.,  1.], [-1.,  1.])




```python
p3.roots()
```




    array([ 1.,  3.])




```python
p2
```




    Polynomial([-2.,  1.], [-1.,  1.], [-1.,  1.])




```python
c1 = P.Chebyshev([1, 2, 3])
```


```python
c1
```




    Chebyshev([ 1.,  2.,  3.], [-1,  1], [-1,  1])




```python
c1.roots()
```




    array([-0.76759188,  0.43425855])




```python
c = P.Chebyshev.fromroots([-1, 1])
```


```python
c
```




    Chebyshev([-0.5,  0. ,  0.5], [-1.,  1.], [-1.,  1.])




```python
l = P.Legendre.fromroots([-1, 1])
```


```python
l
```




    Legendre([-0.66666667,  0.        ,  0.66666667], [-1.,  1.], [-1.,  1.])




```python
c([0.5, 1.5, 2.5])
```




    array([-0.75,  1.25,  5.25])




```python
l([0.5, 1.5, 2.5])
```




    array([-0.75,  1.25,  5.25])



# Polynomial interpolation


```python
x = np.array([1, 2, 3, 4])
y = np.array([1, 3, 5, 4])
```


```python
deg = len(x) - 1
```


```python
A = P.polynomial.polyvander(x, deg)
```


```python
c = linalg.solve(A, y)
```


```python
c
```




    array([ 2. , -3.5,  3. , -0.5])




```python
f1 = P.Polynomial(c)
```


```python
f1(2.5)
```




    4.1875




```python
A = P.chebyshev.chebvander(x, deg)
```


```python
c = linalg.solve(A, y)
```


```python
c
```




    array([ 3.5  , -3.875,  1.5  , -0.125])




```python
f2 = P.Chebyshev(c)
```


```python
f2(2.5)
```




    4.1875




```python
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
```


![png](Ch07_Interpolation_files/Ch07_Interpolation_53_0.png)



```python
f1b = P.Polynomial.fit(x, y, deg)
```


```python
f1b
```




    Polynomial([ 4.1875,  3.1875, -1.6875, -1.6875], [ 1.,  4.], [-1.,  1.])




```python
f2b = P.Chebyshev.fit(x, y, deg)
```


```python
f2b
```




    Chebyshev([ 3.34375 ,  1.921875, -0.84375 , -0.421875], [ 1.,  4.], [-1.,  1.])




```python
np.linalg.cond(P.chebyshev.chebvander(x, deg))
```




    4659.7384241399586




```python
np.linalg.cond(P.chebyshev.chebvander((2*x-5)/3.0, deg))
```




    1.8542033440472896




```python
(2 * x - 5)/3.0
```




    array([-1.        , -0.33333333,  0.33333333,  1.        ])




```python
f1 = P.Polynomial.fit(x, y, 1)
f2 = P.Polynomial.fit(x, y, 2)
f3 = P.Polynomial.fit(x, y, 3)
```


```python
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
```


![png](Ch07_Interpolation_files/Ch07_Interpolation_62_0.png)


## Runge problem


```python
def runge(x):
    return 1/(1 + 25 * x**2)
```


```python
def runge_interpolate(n):
    x = np.linspace(-1, 1, n+1)
    p = P.Polynomial.fit(x, runge(x), deg=n)
    return x, p
```


```python
xx = np.linspace(-1, 1, 250)
```


```python
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
```


![png](Ch07_Interpolation_files/Ch07_Interpolation_67_0.png)


# Spline interpolation


```python
x = np.linspace(-1, 1, 11)
```


```python
y = runge(x)
```


```python
f = interpolate.interp1d(x, y, kind=3)
```


```python
xx = np.linspace(-1, 1, 100)
```


```python
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
```


![png](Ch07_Interpolation_files/Ch07_Interpolation_73_0.png)



```python
x = np.array([0, 1, 2, 3, 4, 5, 6, 7])
```


```python
y = np.array([3, 4, 3.5, 2, 1, 1.5, 1.25, 0.9])
```


```python
xx = np.linspace(x.min(), x.max(), 100)
```


```python
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
```


![png](Ch07_Interpolation_files/Ch07_Interpolation_77_0.png)


# Multivariate interpolation

### Regular grid


```python
x = y = np.linspace(-2, 2, 10)
```


```python
def f(x, y):
    return np.exp(-(x + .5)**2 - 2*(y + .5)**2) - np.exp(-(x - .5)**2 - 2*(y - .5)**2)
```


```python
X, Y = np.meshgrid(x, y)
```


```python
# simulate noisy data at fixed grid points X, Y
Z = f(X, Y) + 0.05 * np.random.randn(*X.shape)
```


```python
f_interp = interpolate.interp2d(x, y, Z, kind='cubic')
```


```python
xx = yy = np.linspace(x.min(), x.max(), 100)
```


```python
ZZi = f_interp(xx, yy)
```


```python
XX, YY = np.meshgrid(xx, yy)
```


```python
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
```


![png](Ch07_Interpolation_files/Ch07_Interpolation_88_0.png)



```python
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
```


![png](Ch07_Interpolation_files/Ch07_Interpolation_89_0.png)


### Irregular grid


```python
np.random.seed(115925231)
```


```python
x = y = np.linspace(-1, 1, 100)
```


```python
X, Y = np.meshgrid(x, y)
```


```python
def f(x, y):
    return np.exp(-x**2 - y**2) * np.cos(4*x) * np.sin(6*y)
```


```python
Z = f(X, Y)
```


```python
N = 500
```


```python
xdata = np.random.uniform(-1, 1, N)
```


```python
ydata = np.random.uniform(-1, 1, N)
```


```python
zdata = f(xdata, ydata)
```


```python
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
```


![png](Ch07_Interpolation_files/Ch07_Interpolation_100_0.png)



```python
def z_interpolate(xdata, ydata, zdata):
    Zi_0 = interpolate.griddata((xdata, ydata), zdata, (X, Y), method='nearest')
    Zi_1 = interpolate.griddata((xdata, ydata), zdata, (X, Y), method='linear')
    Zi_3 = interpolate.griddata((xdata, ydata), zdata, (X, Y), method='cubic')
    return Zi_0, Zi_1, Zi_3
```


```python
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
```


![png](Ch07_Interpolation_files/Ch07_Interpolation_102_0.png)


# Versions


```python
%reload_ext version_information
```


```python
%version_information scipy, numpy, matplotlib
```




<table><tr><th>Software</th><th>Version</th></tr><tr><td>Python</td><td>2.7.10 64bit [GCC 4.2.1 (Apple Inc. build 5577)]</td></tr><tr><td>IPython</td><td>3.2.1</td></tr><tr><td>OS</td><td>Darwin 14.1.0 x86_64 i386 64bit</td></tr><tr><td>scipy</td><td>0.16.0</td></tr><tr><td>numpy</td><td>1.9.2</td></tr><tr><td>matplotlib</td><td>1.4.3</td></tr></table>
