
# Chapter 2 NumPy

<div id="toc"></div>


## 2.1 NumPy Arrays


```python
import numpy as np
# Create an array with 10^7 elements.
arr = np.arange(1e7)
# Converting ndarray to list
larr = arr.tolist()
# Lists cannot by default broadcast,
# so a function is coded to emulate
# what an ndarray can do.

def list_times(alist, scalar):
    for i, val in enumerate(alist):
        alist[i] = val * scalar
        return alist    
```


```python
# Using IPython's magic timeit command
timeit arr * 1.1
# >>> 1 loops, best of 3: 76.9 ms per loop
```


```python
timeit list_times(larr, 1.1)
# >>> 1 loops, best of 3: 2.03 s per loop
```


```python
import numpy as np

# Creating a 3D numpy array
arr = np.zeros((3,3,3))

# Trying to convert array to a matrix, which will not work
mat = np.matrix(arr)

# "ValueError: shape too large to be a matrix."
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-7-319b9cb85414> in <module>()
          3 arr = np.zeros((3,3,3))
          4 # Trying to convert array to a matrix, which will not work
    ----> 5 mat = np.matrix(arr)
          6 # "ValueError: shape too large to be a matrix."
    

    C:\Anaconda\lib\site-packages\numpy\matrixlib\defmatrix.pyc in __new__(subtype, data, dtype, copy)
        256             else:
        257                 intype = N.dtype(dtype)
    --> 258             new = data.view(subtype)
        259             if intype != data.dtype:
        260                 return new.astype(intype)
    

    C:\Anaconda\lib\site-packages\numpy\matrixlib\defmatrix.pyc in __array_finalize__(self, obj)
        301                 return
        302             elif (ndim > 2):
    --> 303                 raise ValueError("shape too large to be a matrix.")
        304         else:
        305             newshape = self.shape
    

    ValueError: shape too large to be a matrix.


### 2.1.1 Array Creation and Data Typing


```python
import numpy as np

# First we create a list and then
# wrap it with the np.array() function.
alist = [1, 2, 3]
arr = np.array(alist)

# Creating an array of zeros with five elements
arr = np.zeros(5)

# What if we want to create an array going from 0 to 100?
arr = np.arange(100)

# Or 10 to 100?
arr = np.arange(10,100)

# If you want 100 steps from 0 to 1...
arr = np.linspace(0, 1, 100)

# Or if you want to generate an array from 1 to 10
# in log10 space in 100 steps...
arr = np.logspace(0, 1, 100, base=10.0)

# Creating a 5x5 array of zeros (an image)
image = np.zeros((5,5))

# Creating a 5x5x5 cube of 1's
# The astype() method sets the array with integer elements.
cube = np.zeros((5,5,5)).astype(int) + 1

# Or even simpler with 16-bit floating-point precision...
cube = np.ones((5, 5, 5)).astype(np.float16)
```


```python
# Array of zero integers
arr = np.zeros(2, dtype=int)

# Array of zero floats
arr = np.zeros(2, dtype=np.float32)
```


```python
# Creating an array with elements from 0 to 999
arr1d = np.arange(1000)

# Now reshaping the array to a 10x10x10 3D array
arr3d = arr1d.reshape((10,10,10))

# The reshape command can alternatively be called this way
arr3d = np.reshape(arr1s, (10, 10, 10))

# Inversely, we can flatten arrays
arr4d = np.zeros((10, 10, 10, 10))
arr1d = arr4d.ravel()

print(arr1d.shape)
(1000,)
```

### 2.1.2 Record Arrays


```python
# Creating an array of zeros and defining column types
recarr = np.zeros((2,), dtype=('i4,f4,a10'))
toadd = [(1,2.,'Hello'),(2,3.,"World")]
recarr[:] = toadd
```


```python
# Creating an array of zeros and defining column types
recarr = np.zeros((2,), dtype=('i4,f4,a10'))

# Now creating the columns we want to put
# in the recarray
col1 = np.arange(2) + 1
col2 = np.arange(2, dtype=np.float32)
col3 = ['Hello', 'World']

# Here we create a list of tuples that is
# identical to the previous toadd list.
toadd = zip(col1, col2, col3)

# Assigning values to recarr
recarr[:] = toadd

# Assigning names to each column, which
# are now by default called 'f0', 'f1', and 'f2'.
recarr.dtype.names = ('Integers' , 'Floats', 'Strings')

# If we want to access one of the columns by its name, we
# can do the following.
recarr('Integers')

# array([1, 2], dtype=int32)
```

1 http://docs.scipy.org/doc/numpy/user/basics.rec.html

### 2.1.3 Indexing and Slicing


```python
alist=[[1,2],[3,4]]

# To return the (0,1) element we must index as shown below.
alist[0][1]
```




    2




```python
# Converting the list defined above into an array
arr = np.array(alist)

# To return the (0,1) element we use ...
arr[0,1]

# Now to access the last column, we simply use ...
arr[:,1]

# Accessing the columns is achieved in the same way,
# which is the bottom row.
arr[1,:]

# Creating an array
arr = np.arange(5)
# Creating the index array
index = np.where(arr > 2)
print(index)
# (array([3, 4]),)
# Creating the desired array
new_arr = arr[index]
```

    (array([3, 4], dtype=int64),)
    


```python
# We use the previous array
new_arr = np.delete(arr, index)
```


```python
index = arr > 2
print(index)
#     [False False True True True]
new_arr = arr[index]
```

    [False False False  True  True]
    

2 http://atpy.github.com

## 2.2 Boolean Statements and NumPy Arrays


```python
# Creating an image
img1 = np.zeros((20, 20)) + 3
img1[4:-4, 4:-4] = 6
img1[7:-7, 7:-7] = 9
# See Plot A
# Let's filter out all values larger than 2 and less than 6.
index1 = img1 > 2
index2 = img1 < 6
compound_index = index1 & index2
# The compound statement can alternatively be written as
compound_index = (img1 > 3) & (img1 < 7)
img2 = np.copy(img1)
img2[compound_index] = 0
# See Plot B.
# Making the boolean arrays even more complex
index3 = img1 == 9
index4 = (index1 & index2) | index3
img3 = np.copy(img1)
img3[index4] = 0
# See Plot C.
```


```python
import numpy as np
import numpy.random as rand
# Creating a 100-element array with random values
# from a standard normal distribution or, in other
# words, a Gaussian distribution.
# The sigma is 1 and the mean is 0.
a = rand.randn(100)
# Here we generate an index for filtering
# out undesired elements.
index = a > 0.2
b = a[index]
# We execute some operation on the desired elements.
b = b ** 2 - 2
# Then we put the modified elements back into the
# original array.
a[index] = b
```

## 2.3 Read and Write

### 2.3.1 Text Files


```python
# Opening the text file with the 'r' option,
# which only allows reading capability
f = open('somefile.txt', 'r')
# Parsing the file and splitting each line,
# which creates a list where each element of
# it is one line
alist = f.readlines()
# Closing file
f.close()
...

# After a few operations, we open a new text file
# to write the data with the 'w' option. If there
# was data already existing in the file, it will be overwritten.
f = open('newtextfile.txt', 'w')
# Writing data to file
f.writelines(newdata)
# Closing file
f.close()
```


```python
import numpy as np
arr = np.loadtxt('somefile.txt')
np.savetxt('somenewfile.txt')
```


```python
# example.txt file looks like the following
#
# XR21 32.789 1
# XR22 33.091 2
table = np.loadtxt('example.txt',
                   dtype='names': ('ID', 'Result', 'Type'),
                   'formats': ('S4', 'f4', 'i2'))
# array([('XR21', 32.78900146484375, 1),
# ('XR22', 33.090999603271484, 2)],
# dtype=[('ID', '|S4'), ('Result', '<f4'), ('Type', '<i2')])
```

### 2.3.2 Binary Files


```python
import numpy as np
# Creating a large array
data = np.empty((1000, 1000))

# Saving the array with numpy.save
np.save('test.npy', data)
# If space is an issue for large files, then
# use numpy.savez instead. It is slower than
# numpy.save because it compresses the binary
# file.

np.savez('test.npz', data)
# Loading the data array
newdata = np.load('test.npy')
```

3 http://matplotlib.sourceforge.net/api/mlab_api.html  
4 http://cxc.harvard.edu/contrib/asciitable/  

## 2.4 Math

### 2.4.1 Linear Algebra


```python
import numpy as np

# Defining the matrices
A = np.matrix([[3, 6, -5],
               [1, -3, 2],
               [5, -1, 4]])
B = np.matrix([[12],
               [-2],
               [10]])

# Solving for the variables, where we invert A
X = A ** (-1) * B
print(X)

# matrix([[ 1.75],
# [ 1.75],
# [ 0.75]])
```

    [[ 1.75]
     [ 1.75]
     [ 0.75]]
    


```python
import numpy as np
a = np.array([[3, 6, -5],
              [1, -3, 2],
              [5, -1, 4]])

# Defining the array
b = np.array([12, -2, 10])

# Solving for the variables, where we invert A
x = np.linalg.inv(a).dot(b)

print(x)
# array([ 1.75, 1.75, 0.75])
```

    [ 1.75  1.75  0.75]
    

5 http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
