
# Chapter 2: Vectors, matrices and multidimensional arrays
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Chapter 2: Vectors, matrices and multidimensional arrays](#chapter-2-vectors-matrices-and-multidimensional-arrays)
  * [The NumPy array object](#the-numpy-array-object)
  * [Data types](#data-types)
    * [Real and imaginary parts](#real-and-imaginary-parts)
  * [Creating arrays](#creating-arrays)
    * [Arrays created from lists and other array-like objects](#arrays-created-from-lists-and-other-array-like-objects)
    * [Arrays filled with constant values](#arrays-filled-with-constant-values)
    * [Arrays filled with incremental sequences](#arrays-filled-with-incremental-sequences)
    * [Arrays filled with logarithmic sequences](#arrays-filled-with-logarithmic-sequences)
    * [Mesh-grid arrays](#mesh-grid-arrays)
    * [Creating uninitialized arrays](#creating-uninitialized-arrays)
    * [Creating arrays with properties of other arrays](#creating-arrays-with-properties-of-other-arrays)
    * [Creating matrix arrays](#creating-matrix-arrays)
  * [Index and slicing](#index-and-slicing)
    * [One-dimensional arrays](#one-dimensional-arrays)
  * [Multidimensional arrays](#multidimensional-arrays)
    * [Views](#views)
    * [Fancy indexing and Boolean-valued indexing](#fancy-indexing-and-boolean-valued-indexing)
  * [Reshaping and resizing](#reshaping-and-resizing)
  * [Vectorized expressions](#vectorized-expressions)
    * [Arithmetic operations](#arithmetic-operations)
    * [Elementwise functions](#elementwise-functions)
    * [Aggregate functions](#aggregate-functions)
    * [Boolean arrays and conditional expressions](#boolean-arrays-and-conditional-expressions)
* [Set operations](#set-operations)
    * [Operations on arrays](#operations-on-arrays)
  * [Matrix and vector operations](#matrix-and-vector-operations)
* [Versions](#versions)

<!-- tocstop -->


---

Robert Johansson

Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).

The source code listings can be downloaded from http://www.apress.com/9781484205549


```python
import numpy as np
```

## The NumPy array object


```python
data = np.array([[1, 2], [3, 4], [5, 6]])
```


```python
type(data)
```




    numpy.ndarray




```python
data
```




    array([[1, 2],
           [3, 4],
           [5, 6]])




```python
data.ndim
```




    2




```python
data.shape
```




    (3, 2)




```python
data.size
```




    6




```python
data.dtype
```




    dtype('int64')




```python
data.nbytes
```




    48



## Data types


```python
np.array([1, 2, 3], dtype=np.int)
```




    array([1, 2, 3])




```python
np.array([1, 2, 3], dtype=np.float)
```




    array([ 1.,  2.,  3.])




```python
np.array([1, 2, 3], dtype=np.complex)
```




    array([ 1.+0.j,  2.+0.j,  3.+0.j])




```python
data = np.array([1, 2, 3], dtype=np.float)
```


```python
data
```




    array([ 1.,  2.,  3.])




```python
data.dtype
```




    dtype('float64')




```python
data = np.array([1, 2, 3], dtype=np.int)
```


```python
data.dtype
```




    dtype('int64')




```python
data
```




    array([1, 2, 3])




```python
data = np.array([1, 2, 3], dtype=np.float)
```


```python
data
```




    array([ 1.,  2.,  3.])




```python
data.astype(np.int)
```




    array([1, 2, 3])




```python
d1 = np.array([1, 2, 3], dtype=float)
```


```python
d2 = np.array([1, 2, 3], dtype=complex)
```


```python
d1 + d2
```




    array([ 2.+0.j,  4.+0.j,  6.+0.j])




```python
(d1 + d2).dtype
```




    dtype('complex128')




```python
np.sqrt(np.array([-1, 0, 1]))
```

    /Users/rob/miniconda/envs/py27-npm-n2/lib/python2.7/site-packages/ipykernel/__main__.py:1: RuntimeWarning: invalid value encountered in sqrt
      if __name__ == '__main__':





    array([ nan,   0.,   1.])




```python
np.sqrt(np.array([-1, 0, 1], dtype=complex))
```




    array([ 0.+1.j,  0.+0.j,  1.+0.j])



### Real and imaginary parts


```python
data = np.array([1, 2, 3], dtype=complex)
```


```python
data
```




    array([ 1.+0.j,  2.+0.j,  3.+0.j])




```python
data.real
```




    array([ 1.,  2.,  3.])




```python
data.imag
```




    array([ 0.,  0.,  0.])



## Creating arrays

### Arrays created from lists and other array-like objects


```python
np.array([1, 2, 3, 4])
```




    array([1, 2, 3, 4])




```python
data.ndim
```




    1




```python
data.shape
```




    (3,)




```python
np.array([[1, 2], [3, 4]])
```




    array([[1, 2],
           [3, 4]])




```python
data.ndim
```




    1




```python
data.shape
```




    (3,)



### Arrays filled with constant values


```python
np.zeros((2, 3))
```




    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])




```python
np.ones(4)
```




    array([ 1.,  1.,  1.,  1.])




```python
data = np.ones(4)
```


```python
data.dtype
```




    dtype('float64')




```python
data = np.ones(4, dtype=np.int64)
```


```python
data.dtype
```




    dtype('int64')




```python
x1 = 5.4 * np.ones(10)
```


```python
x2 = np.full(10, 5.4)
```


```python
x1 = np.empty(5)
```


```python
x1.fill(3.0)
```


```python
x1
```




    array([ 3.,  3.,  3.,  3.,  3.])




```python
x2 = np.full(5, 3.0)
```


```python
x2
```




    array([ 3.,  3.,  3.,  3.,  3.])



### Arrays filled with incremental sequences


```python
np.arange(0.0, 10, 1)
```




    array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])




```python
np.linspace(0, 10, 11)
```




    array([  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.])



### Arrays filled with logarithmic sequences


```python
np.logspace(0, 2, 5)  # 5 data points between 10**0=1 to 10**2=100
```




    array([   1.        ,    3.16227766,   10.        ,   31.6227766 ,  100.        ])



### Mesh-grid arrays


```python
x = np.array([-1, 0, 1])
```


```python
y = np.array([-2, 0, 2])
```


```python
X, Y = np.meshgrid(x, y)
```


```python
X
```




    array([[-1,  0,  1],
           [-1,  0,  1],
           [-1,  0,  1]])




```python
Y
```




    array([[-2, -2, -2],
           [ 0,  0,  0],
           [ 2,  2,  2]])




```python
Z = (X + Y) ** 2
```


```python
Z
```




    array([[9, 4, 1],
           [1, 0, 1],
           [1, 4, 9]])



### Creating uninitialized arrays


```python
np.empty(3, dtype=np.float)
```




    array([ 0.,  0.,  0.])



### Creating arrays with properties of other arrays


```python
def f(x):
    y = np.ones_like(x)
    # compute with x and y
    return y
```

### Creating matrix arrays


```python
np.identity(4)
```




    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])




```python
np.eye(3, k=1)
```




    array([[ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  0.,  0.]])




```python
np.eye(3, k=-1)
```




    array([[ 0.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  1.,  0.]])




```python
np.diag(np.arange(0, 20, 5))
```




    array([[ 0,  0,  0,  0],
           [ 0,  5,  0,  0],
           [ 0,  0, 10,  0],
           [ 0,  0,  0, 15]])



## Index and slicing

### One-dimensional arrays


```python
a = np.arange(0, 11)
```


```python
a
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])




```python
a[0]  # the first element
```




    0




```python
a[-1] # the last element
```




    10




```python
a[4]  # the fifth element, at index 4
```




    4




```python
a[1:-1]
```




    array([1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
a[1:-1:2]
```




    array([1, 3, 5, 7, 9])




```python
a[:5]
```




    array([0, 1, 2, 3, 4])




```python
a[-5:]
```




    array([ 6,  7,  8,  9, 10])




```python
a[::-2]

```




    array([10,  8,  6,  4,  2,  0])



## Multidimensional arrays


```python
f = lambda m, n: n + 10 * m
```


```python
A = np.fromfunction(f, (6, 6), dtype=int)
```


```python
A
```




    array([[ 0,  1,  2,  3,  4,  5],
           [10, 11, 12, 13, 14, 15],
           [20, 21, 22, 23, 24, 25],
           [30, 31, 32, 33, 34, 35],
           [40, 41, 42, 43, 44, 45],
           [50, 51, 52, 53, 54, 55]])




```python
A[:, 1]  # the second column
```




    array([ 1, 11, 21, 31, 41, 51])




```python
A[1, :]  # the second row
```




    array([10, 11, 12, 13, 14, 15])




```python
A[:3, :3]  # upper half diagonal block matrix
```




    array([[ 0,  1,  2],
           [10, 11, 12],
           [20, 21, 22]])




```python
A[3:, :3]  # lower left off-diagonal block matrix
```




    array([[30, 31, 32],
           [40, 41, 42],
           [50, 51, 52]])




```python
A[::2, ::2]  # every second element starting from 0, 0
```




    array([[ 0,  2,  4],
           [20, 22, 24],
           [40, 42, 44]])




```python
A[1::2, 1::3]  # every second element starting from 1, 1
```




    array([[11, 14],
           [31, 34],
           [51, 54]])



### Views


```python
B = A[1:5, 1:5]
```


```python
B
```




    array([[11, 12, 13, 14],
           [21, 22, 23, 24],
           [31, 32, 33, 34],
           [41, 42, 43, 44]])




```python
B[:, :] = 0
```


```python
A
```




    array([[ 0,  1,  2,  3,  4,  5],
           [10,  0,  0,  0,  0, 15],
           [20,  0,  0,  0,  0, 25],
           [30,  0,  0,  0,  0, 35],
           [40,  0,  0,  0,  0, 45],
           [50, 51, 52, 53, 54, 55]])




```python
C = B[1:3, 1:3].copy()
```


```python
C
```




    array([[0, 0],
           [0, 0]])




```python
C[:, :] = 1  # this does not affect B since C is a copy of the view B[1:3, 1:3]
```


```python
C
```




    array([[1, 1],
           [1, 1]])




```python
B
```




    array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])



### Fancy indexing and Boolean-valued indexing


```python
A = np.linspace(0, 1, 11)
```


```python
A[np.array([0, 2, 4])]
```




    array([ 0. ,  0.2,  0.4])




```python
A[[0, 2, 4]]
```




    array([ 0. ,  0.2,  0.4])




```python
A > 0.5
```




    array([False, False, False, False, False, False,  True,  True,  True,
            True,  True], dtype=bool)




```python
A[A > 0.5]
```




    array([ 0.6,  0.7,  0.8,  0.9,  1. ])




```python
A = np.arange(10)
```


```python
A
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
indices = [2, 4, 6]
```


```python
B = A[indices]
```


```python
B[0] = -1  # this does not affect A
```


```python
A
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
A[indices] = -1
```


```python
A
```




    array([ 0,  1, -1,  3, -1,  5, -1,  7,  8,  9])




```python
A = np.arange(10)
```


```python
B = A[A > 5]
```


```python
B[0] = -1  # this does not affect A
```


```python
A
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
A[A > 5] = -1
```


```python
A
```




    array([ 0,  1,  2,  3,  4,  5, -1, -1, -1, -1])



## Reshaping and resizing


```python
data = np.array([[1, 2], [3, 4]])
```


```python
np.reshape(data, (1, 4))
```




    array([[1, 2, 3, 4]])




```python
data.reshape(4)
```




    array([1, 2, 3, 4])




```python
data = np.array([[1, 2], [3, 4]])
```


```python
data
```




    array([[1, 2],
           [3, 4]])




```python
data.flatten()
```




    array([1, 2, 3, 4])




```python
data.flatten().shape
```




    (4,)




```python
data = np.arange(0, 5)
```


```python
column = data[:, np.newaxis]
```


```python
column
```




    array([[0],
           [1],
           [2],
           [3],
           [4]])




```python
row = data[np.newaxis, :]
```


```python
row
```




    array([[0, 1, 2, 3, 4]])




```python
data = np.arange(5)
```


```python
data
```




    array([0, 1, 2, 3, 4])




```python
np.vstack((data, data, data))
```




    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])




```python
data = np.arange(5)
```


```python
data
```




    array([0, 1, 2, 3, 4])




```python
np.hstack((data, data, data))
```




    array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])




```python
data = data[:, np.newaxis]
```


```python
np.hstack((data, data, data))
```




    array([[0, 0, 0],
           [1, 1, 1],
           [2, 2, 2],
           [3, 3, 3],
           [4, 4, 4]])



## Vectorized expressions

### Arithmetic operations


```python
x = np.array([[1, 2], [3, 4]])
```


```python
y = np.array([[5, 6], [7, 8]])
```


```python
x + y
```




    array([[ 6,  8],
           [10, 12]])




```python
y - x
```




    array([[4, 4],
           [4, 4]])




```python
x * y
```




    array([[ 5, 12],
           [21, 32]])




```python
y / x
```




    array([[5, 3],
           [2, 2]])




```python
x * 2
```




    array([[2, 4],
           [6, 8]])




```python
2 ** x
```




    array([[ 2,  4],
           [ 8, 16]])




```python
y / 2
```




    array([[2, 3],
           [3, 4]])




```python
(y / 2).dtype
```




    dtype('int64')




```python
x = np.array([1, 2, 3, 4]).reshape(2,2)
```


```python
z = np.array([1, 2, 3, 4])
```


```python
x / z
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-146-b88ced08eb6a> in <module>()
    ----> 1 x / z


    ValueError: operands could not be broadcast together with shapes (2,2) (4,)



```python
z = np.array([[2, 4]])
```


```python
z
```




    array([[2, 4]])




```python
z.shape
```




    (1, 2)




```python
x / z
```




    array([[0, 0],
           [1, 1]])




```python
zz = np.concatenate([z, z], axis=0)
```


```python
zz
```




    array([[2, 4],
           [2, 4]])




```python
x / zz
```




    array([[0, 0],
           [1, 1]])




```python
z = np.array([[2], [4]])
```


```python
z.shape
```




    (2, 1)




```python
x / z
```




    array([[0, 1],
           [0, 1]])




```python
zz = np.concatenate([z, z], axis=1)
```


```python
zz
```




    array([[2, 2],
           [4, 4]])




```python
x / zz
```




    array([[0, 1],
           [0, 1]])




```python
x = np.array([[1, 3], [2, 4]])
x = x + y
x
```




    array([[ 6,  9],
           [ 9, 12]])




```python
x = np.array([[1, 3], [2, 4]])
x += y
x
```




    array([[ 6,  9],
           [ 9, 12]])



### Elementwise functions


```python
x = np.linspace(-1, 1, 11)
```


```python
x
```




    array([-1. , -0.8, -0.6, -0.4, -0.2,  0. ,  0.2,  0.4,  0.6,  0.8,  1. ])




```python
y = np.sin(np.pi * x)
```


```python
np.round(y, decimals=4)
```




    array([-0.    , -0.5878, -0.9511, -0.9511, -0.5878,  0.    ,  0.5878,
            0.9511,  0.9511,  0.5878,  0.    ])




```python
np.add(np.sin(x) ** 2, np.cos(x) ** 2)
```




    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])




```python
np.sin(x) ** 2 + np.cos(x) ** 2
```




    array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])




```python
def heaviside(x):
    return 1 if x > 0 else 0
```


```python
heaviside(-1)
```




    0




```python
heaviside(1.5)
```




    1




```python
x = np.linspace(-5, 5, 11)
```


```python
heaviside(x)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-172-fe9fb94bcf99> in <module>()
    ----> 1 heaviside(x)


    <ipython-input-168-01b8e2f570bc> in heaviside(x)
          1 def heaviside(x):
    ----> 2     return 1 if x > 0 else 0


    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()



```python
heaviside = np.vectorize(heaviside)
```


```python
heaviside(x)
```




    array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])




```python
def heaviside(x):
    return 1.0 * (x > 0)
```

### Aggregate functions


```python
data = np.random.normal(size=(15,15))
```


```python
np.mean(data)
```




    0.081598599126035468




```python
data.mean()
```




    0.081598599126035468




```python
data = np.random.normal(size=(5, 10, 15))
```


```python
data.sum(axis=0).shape
```




    (10, 15)




```python
data.sum(axis=(0, 2)).shape
```




    (10,)




```python
data.sum()
```




    -11.37821321738001




```python
data = np.arange(1,10).reshape(3,3)
```


```python
data
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
data.sum()
```




    45




```python
data.sum(axis=0)
```




    array([12, 15, 18])




```python
data.sum(axis=1)
```




    array([ 6, 15, 24])



### Boolean arrays and conditional expressions


```python
a = np.array([1, 2, 3, 4])
```


```python
b = np.array([4, 3, 2, 1])
```


```python
a < b
```




    array([ True,  True, False, False], dtype=bool)




```python
np.all(a < b)
```




    False




```python
np.any(a < b)
```




    True




```python
if np.all(a < b):
    print("All elements in a are smaller than their corresponding element in b")
elif np.any(a < b):
    print("Some elements in a are smaller than their corresponding elemment in b")
else:
    print("All elements in b are smaller than their corresponding element in a")
```

    Some elements in a are smaller than their corresponding elemment in b



```python
x = np.array([-2, -1, 0, 1, 2])
```


```python
x > 0
```




    array([False, False, False,  True,  True], dtype=bool)




```python
1 * (x > 0)
```




    array([0, 0, 0, 1, 1])




```python
x * (x > 0)
```




    array([0, 0, 0, 1, 2])




```python
def pulse(x, position, height, width):
    return height * (x >= position) * (x <= (position + width))
```


```python
x = np.linspace(-5, 5, 11)
```


```python
pulse(x, position=-2, height=1, width=5)
```




    array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0])




```python
pulse(x, position=1, height=1, width=5)
```




    array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])




```python
def pulse(x, position, height, width):
    return height * np.logical_and(x >= position, x <= (position + width))
```


```python
x = np.linspace(-4, 4, 9)
```


```python
np.where(x < 0, x**2, x**3)
```




    array([ 16.,   9.,   4.,   1.,   0.,   1.,   8.,  27.,  64.])




```python
np.select([x < -1, x < 2, x >= 2],
          [x**2  , x**3 , x**4])
```




    array([  16.,    9.,    4.,   -1.,    0.,    1.,   16.,   81.,  256.])




```python
np.choose([0, 0, 0, 1, 1, 1, 2, 2, 2],
          [x**2,    x**3,    x**4])
```




    array([  16.,    9.,    4.,   -1.,    0.,    1.,   16.,   81.,  256.])




```python
x[abs(x) > 2]
```




    array([-4., -3.,  3.,  4.])




```python
np.nonzero(abs(x) > 2)
```




    (array([0, 1, 7, 8]),)




```python
x[np.nonzero(abs(x) > 2)]
```




    array([-4., -3.,  3.,  4.])



# Set operations


```python
a = np.unique([1,2,3,3])
```


```python
b = np.unique([2,3,4,4,5,6,5])
```


```python
np.in1d(a, b)
```




    array([False,  True,  True], dtype=bool)




```python
1 in a
```




    True




```python
1 in b
```




    False




```python
np.all(np.in1d(a, b))
```




    False




```python
np.union1d(a, b)
```




    array([1, 2, 3, 4, 5, 6])




```python
np.intersect1d(a, b)
```




    array([2, 3])




```python
np.setdiff1d(a, b)
```




    array([1])




```python
np.setdiff1d(b, a)
```




    array([4, 5, 6])



### Operations on arrays


```python
data = np.arange(9).reshape(3, 3)
```


```python
data
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
np.transpose(data)
```




    array([[0, 3, 6],
           [1, 4, 7],
           [2, 5, 8]])




```python
data = np.random.randn(1, 2, 3, 4, 5)
```


```python
data.shape
```




    (1, 2, 3, 4, 5)




```python
data.T.shape
```




    (5, 4, 3, 2, 1)



## Matrix and vector operations


```python
A = np.arange(1, 7).reshape(2, 3)
```


```python
A
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
B = np.arange(1, 7).reshape(3, 2)
```


```python
B
```




    array([[1, 2],
           [3, 4],
           [5, 6]])




```python
np.dot(A, B)
```




    array([[22, 28],
           [49, 64]])




```python
np.dot(B, A)
```




    array([[ 9, 12, 15],
           [19, 26, 33],
           [29, 40, 51]])




```python
A = np.arange(9).reshape(3, 3)
```


```python
A
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
x = np.arange(3)
```


```python
x
```




    array([0, 1, 2])




```python
np.dot(A, x)
```




    array([ 5, 14, 23])




```python
A.dot(x)
```




    array([ 5, 14, 23])




```python
A = np.random.rand(3,3)
B = np.random.rand(3,3)
```


```python
Ap = np.dot(B, np.dot(A, np.linalg.inv(B)))
```


```python
Ap = B.dot(A.dot(np.linalg.inv(B)))
```


```python
A = np.matrix(A)
```


```python
B = np.matrix(B)
```


```python
Ap = B * A * B.I
```


```python
A = np.asmatrix(A)
```


```python
B = np.asmatrix(B)
```


```python
Ap = B * A * B.I
```


```python
Ap = np.asarray(Ap)
```


```python
np.inner(x, x)
```




    5




```python
np.dot(x, x)
```




    5




```python
y = x[:, np.newaxis]
```


```python
y
```




    array([[0],
           [1],
           [2]])




```python
np.dot(y.T, y)
```




    array([[5]])




```python
x = np.array([1, 2, 3])
```


```python
np.outer(x, x)
```




    array([[1, 2, 3],
           [2, 4, 6],
           [3, 6, 9]])




```python
np.kron(x, x)
```




    array([1, 2, 3, 2, 4, 6, 3, 6, 9])




```python
np.kron(x[:, np.newaxis], x[np.newaxis, :])
```




    array([[1, 2, 3],
           [2, 4, 6],
           [3, 6, 9]])




```python
np.kron(np.ones((2,2)), np.identity(2))
```




    array([[ 1.,  0.,  1.,  0.],
           [ 0.,  1.,  0.,  1.],
           [ 1.,  0.,  1.,  0.],
           [ 0.,  1.,  0.,  1.]])




```python
np.kron(np.identity(2), np.ones((2,2)))
```




    array([[ 1.,  1.,  0.,  0.],
           [ 1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  1.],
           [ 0.,  0.,  1.,  1.]])




```python
x = np.array([1, 2, 3, 4])
```


```python
y = np.array([5, 6, 7, 8])
```


```python
np.einsum("n,n", x, y)
```




    70




```python
np.inner(x, y)
```




    70




```python
A = np.arange(9).reshape(3, 3)
```


```python
B = A.T
```


```python
np.einsum("mk,kn", A, B)
```




    array([[  5,  14,  23],
           [ 14,  50,  86],
           [ 23,  86, 149]])




```python
np.alltrue(np.einsum("mk,kn", A, B) == np.dot(A, B))
```




    True



# Versions


```python
%reload_ext version_information
%version_information numpy
```




<table><tr><th>Software</th><th>Version</th></tr><tr><td>Python</td><td>2.7.10 64bit [GCC 4.2.1 (Apple Inc. build 5577)]</td></tr><tr><td>IPython</td><td>4.0.0</td></tr><tr><td>OS</td><td>Darwin 14.5.0 x86_64 i386 64bit</td></tr><tr><td>numpy</td><td>1.9.2</td></tr></table>
