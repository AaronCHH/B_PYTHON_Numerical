
# Chapter 1: Computing with Python
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->
<!-- tocstop -->


---

Robert Johansson

Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).

The source code listings can be downloaded from http://www.apress.com/9781484205549

## Interpreter


```python
%%writefile hello.py
print("Hello from Python!")
```

    Overwriting hello.py



```python
!python hello.py
```

    Hello from Python!



```python
!python27 --version
```

    Python 2.7.13 :: Anaconda custom (64-bit)


## Input and output caching


```python
3 * 3
```




    9




```python
In[1]
```




    u'3 * 3'




```python
Out[1]
```




    9




```python
In
```




    ['', u'3 * 3', u'In[1]', u'Out[1]', u'In']




```python
Out
```




    {1: 9,
     2: u'3 * 3',
     3: 9,
     4: ['', u'3 * 3', u'In[1]', u'Out[1]', u'In', u'Out']}




```python
1+2
```




    3




```python
1+2;
```


```python
x = 1
```


```python
x = 2; x
```




    2



## Documentation


```python
import os
```


```python
# try os.w<TAB>
```


```python
import math
```


```python
math.cos?
```

## Interaction with System Shell


```python
!touch file1.py file2.py file3.py
```





```python
!ls file*
```


```python
files = !ls file*
```


```python
len(files)
```




    3




```python
files
```




    ['file1.py', 'file2.py', 'file3.py']




```python
file = "file1.py"
```


```python
!ls -l $file
```

    -rw-r--r--  1 rob  staff  0 Aug 30 17:03 file1.py



## Running scripts from the IPython console


```python
%%writefile fib.py

def fib(N):
    """
    Return a list of the first N Fibonacci numbers.
    """
    f0, f1 = 0, 1
    f = [1] * N
    for n in range(1, N):
        f[n] = f0 + f1
        f0, f1 = f1, f[n]

    return f

print(fib(10))
```

    Overwriting fib.py



```python
!python fib.py
```

    [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]




```python
%run fib.py
```

    [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]



```python
fib(6)
```




    [1, 1, 2, 3, 5, 8]



## Debugger


```python
fib(1.0)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-25-ccc1774a65b9> in <module>()
    ----> 1 fib(1.0)


    /Users/rob/Desktop/apress-numerical-python-review/code/fib.py in fib(N)
          5     """
          6     f0, f1 = 0, 1
    ----> 7     f = [1] * N
          8     for n in range(1, N):
          9         f[n] = f0 + f1


    TypeError: can't multiply sequence by non-int of type 'float'



```python
%debug
```

    > [0;32m/Users/rob/Desktop/apress-numerical-python-review/code/fib.py[0m(7)[0;36mfib[0;34m()[0m
    [0;32m      6 [0;31m    [0mf0[0m[0;34m,[0m [0mf1[0m [0;34m=[0m [0;36m0[0m[0;34m,[0m [0;36m1[0m[0;34m[0m[0m
    [0m[0;32m----> 7 [0;31m    [0mf[0m [0;34m=[0m [0;34m[[0m[0;36m1[0m[0;34m][0m [0;34m*[0m [0mN[0m[0;34m[0m[0m
    [0m[0;32m      8 [0;31m    [0;32mfor[0m [0mn[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m1[0m[0;34m,[0m [0mN[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0m
    [0m
    ipdb> print(N)
    1.0
    ipdb> q


## Timing and profiling code


```python
%timeit fib(100)
```

    100000 loops, best of 3: 18.2 µs per loop



```python
result = %time fib(100)
```

    CPU times: user 38 µs, sys: 22 µs, total: 60 µs
    Wall time: 47 µs



```python
len(result)
```




    100




```python
import numpy as np

def random_walker_max_distance(M, N):
    """
    Simulate N random walkers taking M steps, and return the largest distance
    from the starting point achieved by any of the random walkers.
    """
    trajectories = [np.random.randn(M).cumsum() for _ in range(N)]
    return np.max(np.abs(trajectories))
```


```python
%prun random_walker_max_distance(400, 10000)
```



## IPython nbconvert


```python
!ipython nbconvert --to html ch01-code-listing.ipynb
```

    [NbConvertApp] Converting notebook ch01-code-listing.ipynb to html
    [NbConvertApp] Writing 219405 bytes to ch01-code-listing.html



```python
!ipython nbconvert --to pdf ch01-code-listing.ipynb
```

    [NbConvertApp] Converting notebook ch01-code-listing.ipynb to pdf
    [NbConvertApp] Writing 25971 bytes to notebook.tex
    [NbConvertApp] Building PDF
    [NbConvertApp] Running pdflatex 3 times: [u'pdflatex', u'notebook.tex']
    [NbConvertApp] PDF successfully created
    [NbConvertApp] Writing 114085 bytes to ch01-code-listing.pdf



```python
%%writefile custom_template.tplx
((*- extends 'article.tplx' -*))

((* block title *)) \title{Document title} ((* endblock title *))
((* block author *)) \author{Author's Name} ((* endblock author *))
```

    Overwriting custom_template.tplx



```python
!ipython nbconvert ch01-code-listing.ipynb --to pdf --template custom_template.tplx
```

    [NbConvertApp] Converting notebook ch01-code-listing.ipynb to pdf
    [NbConvertApp] Writing 25994 bytes to notebook.tex
    [NbConvertApp] Building PDF
    [NbConvertApp] Running pdflatex 3 times: [u'pdflatex', u'notebook.tex']
    [NbConvertApp] PDF successfully created
    [NbConvertApp] Writing 114961 bytes to ch01-code-listing.pdf



```python
!ipython nbconvert ch01-code-listing.ipynb --to python
```

    [NbConvertApp] Converting notebook ch01-code-listing.ipynb to python
    [NbConvertApp] Writing 3237 bytes to ch01-code-listing.py


# Versions


```python
%reload_ext version_information
%version_information numpy
```




<table><tr><th>Software</th><th>Version</th></tr><tr><td>Python</td><td>2.7.10 64bit [GCC 4.2.1 (Apple Inc. build 5577)]</td></tr><tr><td>IPython</td><td>4.0.0</td></tr><tr><td>OS</td><td>Darwin 14.5.0 x86_64 i386 64bit</td></tr><tr><td>numpy</td><td>1.9.2</td></tr></table>

