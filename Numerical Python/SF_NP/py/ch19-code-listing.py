
# coding: utf-8

# # Chapter 19: Code optimization

# Robert Johansson
# 
# Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).
# 
# The source code listings can be downloaded from http://www.apress.com/9781484205549

# In[1]:

import numba


# In[2]:

import pyximport


# In[3]:

import cython


# In[4]:

import numpy as np


# In[5]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# # Numba

# In[6]:

np.random.seed(0)


# In[7]:

data = np.random.randn(50000)


# In[8]:

def py_sum(data):
    s = 0
    for d in data:
        s += d
    return s


# In[9]:

def py_cumsum(data):
    out = np.zeros(len(data), dtype=np.float64)
    s = 0 
    for n in range(len(data)):
        s += data[n]
        out[n] = s

    return out


# In[10]:

get_ipython().magic(u'timeit py_sum(data)')


# In[11]:

assert abs(py_sum(data) - np.sum(data)) < 1e-10


# In[12]:

get_ipython().magic(u'timeit np.sum(data)')


# In[13]:

get_ipython().magic(u'timeit py_cumsum(data)')


# In[14]:

assert np.allclose(np.cumsum(data), py_cumsum(data))


# In[15]:

get_ipython().magic(u'timeit np.cumsum(data)')


# In[16]:

@numba.jit
def jit_sum(data):
    s = 0 
    for d in data:
        s += d

    return s


# In[17]:

assert abs(jit_sum(data) - np.sum(data)) < 1e-10


# In[18]:

get_ipython().magic(u'timeit jit_sum(data)')


# In[19]:

jit_cumsum = numba.jit()(py_cumsum)


# In[20]:

assert np.allclose(np.cumsum(data), jit_cumsum(data))


# In[21]:

get_ipython().magic(u'timeit jit_cumsum(data)')


# ## Julia fractal

# In[22]:

def py_julia_fractal(z_re, z_im, j):
    for m in range(len(z_re)):
        for n in range(len(z_im)):
            z = z_re[m] + 1j * z_im[n]
            for t in range(256):
                z = z ** 2 - 0.05 + 0.68j
                if np.abs(z) > 2.0:
                #if (z.real * z.real + z.imag * z.imag) > 4.0:  # a bit faster
                    j[m, n] = t
                    break


# In[23]:

jit_julia_fractal = numba.jit(nopython=True)(py_julia_fractal)


# In[24]:

N = 1024
j = np.zeros((N, N), np.int64)
z_real = np.linspace(-1.5, 1.5, N)
z_imag = np.linspace(-1.5, 1.5, N)


# In[25]:

jit_julia_fractal(z_real, z_imag, j)


# In[26]:

fig, ax = plt.subplots(figsize=(14, 14))
ax.imshow(j, cmap=plt.cm.RdBu_r,
          extent=[-1.5, 1.5, -1.5, 1.5])
ax.set_xlabel("$\mathrm{Re}(z)$", fontsize=18)
ax.set_ylabel("$\mathrm{Im}(z)$", fontsize=18)
fig.tight_layout()
fig.savefig("ch19-numba-julia-fractal.pdf")


# In[27]:

get_ipython().magic(u'timeit py_julia_fractal(z_real, z_imag, j)')


# In[28]:

get_ipython().magic(u'timeit jit_julia_fractal(z_real, z_imag, j)')


# ## Vectorize

# In[29]:

def py_Heaviside(x):
    if x == 0.0:
        return 0.5
    
    if x < 0.0:
        return 0.0
    else:
        return 1.0


# In[30]:

x = np.linspace(-2, 2, 50001)


# In[31]:

get_ipython().magic(u'timeit [py_Heaviside(xx) for xx in x]')


# In[32]:

np_vec_Heaviside = np.vectorize(py_Heaviside)


# In[33]:

np_vec_Heaviside(x)


# In[34]:

get_ipython().magic(u'timeit np_vec_Heaviside(x)')


# In[35]:

def np_Heaviside(x):
    return (x > 0.0) + (x == 0.0)/2.0


# In[36]:

get_ipython().magic(u'timeit np_Heaviside(x)')


# In[37]:

@numba.vectorize([numba.float32(numba.float32),
                  numba.float64(numba.float64)])
def jit_Heaviside(x):
    if x == 0.0:
        return 0.5
    
    if x < 0:
        return 0.0
    else:
        return 1.0


# In[38]:

get_ipython().magic(u'timeit jit_Heaviside(x)')


# In[39]:

jit_Heaviside([-1, -0.5, 0.0, 0.5, 1.0])


# # Cython

# In[40]:

get_ipython().system(u'rm cy_sum.*')


# In[41]:

get_ipython().run_cell_magic(u'writefile', u'cy_sum.pyx', u'\ndef cy_sum(data):\n    s = 0.0\n    for d in data:\n        s += d\n    return s')


# In[42]:

get_ipython().system(u'cython cy_sum.pyx')


# In[43]:

# 5 lines of python code -> 1470 lines of C code ...
get_ipython().system(u'wc cy_sum.c')


# In[44]:

get_ipython().run_cell_magic(u'writefile', u'setup.py', u"\nfrom distutils.core import setup\nfrom Cython.Build import cythonize\n\nimport numpy as np\nsetup(ext_modules=cythonize('cy_sum.pyx'),\n      include_dirs=[np.get_include()],\n      requires=['Cython', 'numpy'] )")


# In[45]:

get_ipython().system(u'python setup.py build_ext --inplace > /dev/null')


# In[46]:

from cy_sum import cy_sum


# In[47]:

cy_sum(data)


# In[48]:

get_ipython().magic(u'timeit cy_sum(data)')


# In[49]:

get_ipython().magic(u'timeit py_sum(data)')


# In[50]:

get_ipython().run_cell_magic(u'writefile', u'cy_cumsum.pyx', u'\ncimport numpy\nimport numpy\n\ndef cy_cumsum(data):\n    out = numpy.zeros_like(data)\n    s = 0 \n    for n in range(len(data)):\n        s += data[n]\n        out[n] = s\n\n    return out')


# In[51]:

pyximport.install(setup_args={'include_dirs': np.get_include()});


# In[52]:

pyximport.install(setup_args=dict(include_dirs=np.get_include()));


# In[53]:

from cy_cumsum import cy_cumsum


# In[54]:

get_ipython().magic(u'timeit cy_cumsum(data)')


# In[55]:

get_ipython().magic(u'timeit py_cumsum(data)')


# ## Using IPython cython command

# In[56]:

get_ipython().magic(u'load_ext cython')


# In[57]:

get_ipython().run_cell_magic(u'cython', u'-a', u'def cy_sum(data):\n    s = 0.0\n    for d in data:\n        s += d\n    return s')


# In[58]:

get_ipython().magic(u'timeit cy_sum(data)')


# In[59]:

get_ipython().magic(u'timeit py_sum(data)')


# In[60]:

assert np.allclose(np.sum(data), cy_sum(data))


# In[61]:

get_ipython().run_cell_magic(u'cython', u'-a', u'cimport numpy\ncimport cython\n\n@cython.boundscheck(False)\n@cython.wraparound(False)\ndef cy_sum(numpy.ndarray[numpy.float64_t, ndim=1] data):\n    cdef numpy.float64_t s = 0.0\n    cdef int n, N = len(data)\n    for n in range(N):\n        s += data[n]\n    return s')


# In[62]:

get_ipython().magic(u'timeit cy_sum(data)')


# In[63]:

get_ipython().magic(u'timeit jit_sum(data)')


# In[64]:

get_ipython().magic(u'timeit np.sum(data)')


# ## Cummulative sum

# In[65]:

get_ipython().run_cell_magic(u'cython', u'-a', u'cimport numpy\nimport numpy\ncimport cython\n\nctypedef numpy.float64_t FTYPE_t\n\n@cython.boundscheck(False)\n@cython.wraparound(False)\ndef cy_cumsum(numpy.ndarray[FTYPE_t, ndim=1] data):\n    cdef int N = data.size\n    cdef numpy.ndarray[FTYPE_t, ndim=1] out = numpy.zeros(N, dtype=data.dtype)\n    cdef numpy.float64_t s = 0.0\n    for n in range(N):\n        s += data[n]\n        out[n] = s\n    return out')


# In[66]:

get_ipython().magic(u'timeit py_cumsum(data)')


# In[67]:

get_ipython().magic(u'timeit cy_cumsum(data)')


# In[68]:

get_ipython().magic(u'timeit jit_cumsum(data)')


# In[69]:

get_ipython().magic(u'timeit np.cumsum(data)')


# In[70]:

assert np.allclose(cy_cumsum(data), np.cumsum(data))


# ## Fused types

# In[71]:

py_sum([1.0, 2.0, 3.0, 4.0, 5.0])


# In[72]:

py_sum([1, 2, 3, 4, 5])


# In[73]:

cy_sum(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


# In[74]:

cy_sum(np.array([1, 2, 3, 4, 5]))


# In[75]:

get_ipython().run_cell_magic(u'cython', u'-a', u'cimport numpy\ncimport cython\n\nctypedef fused I_OR_F_t:\n    numpy.int64_t \n    numpy.float64_t \n\n@cython.boundscheck(False)\n@cython.wraparound(False)\ndef cy_fused_sum(numpy.ndarray[I_OR_F_t, ndim=1] data):\n    cdef I_OR_F_t s = 0\n    cdef int n, N = data.size\n    for n in range(N):\n        s += data[n]\n    return s')


# In[76]:

cy_fused_sum(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))


# In[77]:

cy_fused_sum(np.array([1, 2, 3, 4, 5]))


# ## Julia fractal

# In[78]:

get_ipython().run_cell_magic(u'cython', u'-a', u'cimport numpy\ncimport cython\n\nctypedef numpy.int64_t ITYPE_t\nctypedef numpy.float64_t FTYPE_t\n\ncpdef inline double abs2(double complex z):\n    return z.real * z.real + z.imag * z.imag\n\n@cython.boundscheck(False)\n@cython.wraparound(False)\ndef cy_julia_fractal(numpy.ndarray[FTYPE_t, ndim=1] z_re, \n                     numpy.ndarray[FTYPE_t, ndim=1] z_im, \n                     numpy.ndarray[ITYPE_t, ndim=2] j):\n    cdef int m, n, t, M = z_re.size, N = z_im.size\n    cdef double complex z\n    for m in range(M):\n        for n in range(N):\n            z = z_re[m] + 1.0j * z_im[n]\n            for t in range(256):\n                z = z ** 2 - 0.05 + 0.68j\n                if abs2(z) > 4.0:\n                    j[m, n] = t\n                    break')


# In[79]:

N = 1024


# In[80]:

j = np.zeros((N, N), dtype=np.int64)


# In[81]:

z_real = np.linspace(-1.5, 1.5, N)


# In[82]:

z_imag = np.linspace(-1.5, 1.5, N)


# In[83]:

get_ipython().magic(u'timeit cy_julia_fractal(z_real, z_imag, j)')


# In[84]:

get_ipython().magic(u'timeit jit_julia_fractal(z_real, z_imag, j)')


# In[85]:

j1 = np.zeros((N, N), dtype=np.int64)


# In[86]:

cy_julia_fractal(z_real, z_imag, j1)


# In[87]:

j2 = np.zeros((N, N), dtype=np.int64)


# In[88]:

jit_julia_fractal(z_real, z_imag, j2)


# In[89]:

assert np.allclose(j1, j2)


# ## Calling C function

# In[90]:

get_ipython().run_cell_magic(u'cython', u'', u'\ncdef extern from "math.h":\n     double acos(double)\n\ndef cy_acos1(double x):\n    return acos(x)')


# In[91]:

get_ipython().magic(u'timeit cy_acos1(0.5)')


# In[92]:

get_ipython().run_cell_magic(u'cython', u'', u'\nfrom libc.math cimport acos\n\ndef cy_acos2(double x):\n    return acos(x)')


# In[93]:

get_ipython().magic(u'timeit cy_acos2(0.5)')


# In[94]:

from numpy import arccos


# In[95]:

get_ipython().magic(u'timeit arccos(0.5)')


# In[96]:

from math import acos


# In[97]:

get_ipython().magic(u'timeit acos(0.5)')


# In[98]:

assert cy_acos1(0.5) == acos(0.5)


# In[99]:

assert cy_acos2(0.5) == acos(0.5)


# # Versions

# In[100]:

get_ipython().magic(u'reload_ext version_information')


# In[101]:

get_ipython().magic(u'version_information numpy, cython, numba, matplotlib')

