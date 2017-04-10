
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

- [2 Systems of Linear Algebraic Equations](#2-systems-of-linear-algebraic-equations)
- [2.1 Introduction](#21-introduction)
	- [Notation](#notation)
	- [Uniqueness of Solution](#uniqueness-of-solution)
	- [Ill Conditioning](#ill-conditioning)
	- [Linear Systems](#linear-systems)
	- [Methods of Solution](#methods-of-solution)
	- [Overview of Direct Methods](#overview-of-direct-methods)
- [2.2 Gauss Elimination Method](#22-gauss-elimination-method)
	- [Introduction](#introduction)
	- [Algorithm for Gauss Elimination Method](#algorithm-for-gauss-elimination-method)
	- [Multiple Sets of Equations](#multiple-sets-of-equations)
- [2.3 LU Decomposition Methods](#23-lu-decomposition-methods)
	- [Introduction](#introduction-1)
	- [Doolittle’s Decomposition Method](#doolittles-decomposition-method)
	- [Choleski’s Decomposition Method](#choleskis-decomposition-method)
	- [Other Methods](#other-methods)
		- [Crout’s decomposition](#crouts-decomposition)
		- [Gauss-Jordan Elimination](#gauss-jordan-elimination)
- [2.4 Symmetric and Banded Coefficient Matrices](#24-symmetric-and-banded-coefficient-matrices)
		- [Introduction](#introduction-2)
		- [Tridiagonal Coefficient Matrix](#tridiagonal-coefficient-matrix)
		- [Symmetric Coefficient Matrices](#symmetric-coefficient-matrices)
		- [Symmetric, Pentadiagonal Coefficient Matrices](#symmetric-pentadiagonal-coefficient-matrices)
- [2.5 Pivoting](#25-pivoting)
	- [Introduction](#introduction-3)
	- [Diagonal Dominance](#diagonal-dominance)
	- [Gauss Elimination with Scaled Row Pivoting](#gauss-elimination-with-scaled-row-pivoting)
		- [When to Pivot](#when-to-pivot)
	- [When to Pivot](#when-to-pivot-1)
- [2.6 Matrix Inversion](#26-matrix-inversion)
- [2.7 Iterative Methods](#27-iterative-methods)
	- [Intrduction](#intrduction)
	- [Gauss-Seidel Method](#gauss-seidel-method)
	- [Conjugate Gradient Method](#conjugate-gradient-method)
- [2.8 Other Methods](#28-other-methods)

<!-- tocstop -->


# 2 Systems of Linear Algebraic Equations

# 2.1 Introduction

## Notation

## Uniqueness of Solution

## Ill Conditioning

## Linear Systems

## Methods of Solution

## Overview of Direct Methods

# 2.2 Gauss Elimination Method

## Introduction

## Algorithm for Gauss Elimination Method


```python
# %load code/gaussElimin.py
## module gaussElimin
''' x = gaussElimin(a,b).
    Solves [a]{b} = {x} by Gauss elimination.
'''
import numpy as np

def gaussElimin(a,b):
    n = len(b)
  # Elimination Phase
    for k in range(0,n-1):
        for i in range(k+1,n):
           if a[i,k] != 0.0:
               lam = a [i,k]/a[k,k]
               a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
               b[i] = b[i] - lam*b[k]
  # Back substitution
    for k in range(n-1,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    return b

```

## Multiple Sets of Equations


```python
#!/usr/bin/python
## example2_4
import numpy as np
from gaussElimin import *
def vandermode(v):
    n = len(v)
    a = np.zeros((n,n))
    for j in range(n):
        a[:,j] = v**(n-j-1)
    return a
v = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
b = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
a = vandermode(v)
aOrig = a.copy() # Save original matrix
bOrig = b.copy() # and the constant vector
x = gaussElimin(a,b)
det = np.prod(np.diagonal(a))
print('x =\n',x)
print('\ndet =',det)
print('\nCheck result: [a]{x} - b =\n',np.dot(aOrig,x) - bOrig)
input("\nPress return to exit")
```

    x =
     [   416.66666667  -3125.00000004   9250.00000012 -13500.00000017
       9709.33333345  -2751.00000003]

    det = -1.13246207999e-06

    Check result: [a]{x} - b =
     [  0.00000000e+00   0.00000000e+00   1.81898940e-12   1.09139364e-11
      -1.81898940e-11   4.36557457e-11]


# 2.3 LU Decomposition Methods

## Introduction

## Doolittle’s Decomposition Method

**Decomposition phase**

**Solution phase**


```python
# %load code/LUdecomp.py
## module LUdecomp
''' a = LUdecomp(a)
    LUdecomposition: [L][U] = [a]

    x = LUsolve(a,b)
    Solution phase: solves [L][U]{x} = {b}
'''
import numpy as np

def LUdecomp(a):
    n = len(a)
    for k in range(0,n-1):
        for i in range(k+1,n):
           if a[i,k] != 0.0:
               lam = a [i,k]/a[k,k]
               a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
               a[i,k] = lam
    return a

def LUsolve(a,b):
    n = len(a)
    for k in range(1,n):
        b[k] = b[k] - np.dot(a[k,0:k],b[0:k])
    b[n-1] = b[n-1]/a[n-1,n-1]    
    for k in range(n-2,-1,-1):
       b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    return b


```

## Choleski’s Decomposition Method


```python
# %load code/choleski.py
## module choleski
''' L = choleski(a)
    Choleski decomposition: [L][L]transpose = [a]

    x = choleskiSol(L,b)
    Solution phase of Choleski's decomposition method
'''
import numpy as np
import math
import error

def choleski(a):
    n = len(a)
    for k in range(n):
        try:
            a[k,k] = math.sqrt(a[k,k] - np.dot(a[k,0:k],a[k,0:k]))
        except ValueError:
            error.err('Matrix is not positive definite')
        for i in range(k+1,n):
            a[i,k] = (a[i,k] - np.dot(a[i,0:k],a[k,0:k]))/a[k,k]
    for k in range(1,n): a[0:k,k] = 0.0
    return a

def choleskiSol(L,b):
    n = len(b)
  # Solution of [L]{y} = {b}  
    for k in range(n):
        b[k] = (b[k] - np.dot(L[k,0:k],b[0:k]))/L[k,k]
  # Solution of [L_transpose]{x} = {y}      
    for k in range(n-1,-1,-1):
        b[k] = (b[k] - np.dot(L[k+1:n,k],b[k+1:n]))/L[k,k]
    return b


```

## Other Methods

### Crout’s decomposition

### Gauss-Jordan Elimination

# 2.4 Symmetric and Banded Coefficient Matrices

### Introduction

### Tridiagonal Coefficient Matrix


```python
# %load code/LUdecomp3.py
## module LUdecomp3
''' c,d,e = LUdecomp3(c,d,e).
    LU decomposition of tridiagonal matrix [a], where {c}, {d}
    and {e} are the diagonals of [a]. On output
    {c},{d} and {e} are the diagonals of the decomposed matrix.

    x = LUsolve3(c,d,e,b).
    Solution of [a]{x} {b}, where {c}, {d} and {e} are the
    vectors returned from LUdecomp3.
'''

def LUdecomp3(c,d,e):
    n = len(d)
    for k in range(1,n):
        lam = c[k-1]/d[k-1]
        d[k] = d[k] - lam*e[k-1]
        c[k-1] = lam
    return c,d,e

def LUsolve3(c,d,e,b):
    n = len(d)
    for k in range(1,n):
        b[k] = b[k] - c[k-1]*b[k-1]
    b[n-1] = b[n-1]/d[n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - e[k]*b[k+1])/d[k]
    return b   



```

### Symmetric Coefficient Matrices

### Symmetric, Pentadiagonal Coefficient Matrices


```python
# %load code/LUdecomp5.py
## module LUdecomp5
''' d,e,f = LUdecomp5(d,e,f).
    LU decomposition of symetric pentadiagonal matrix [a], where
    {f}, {e} and {d} are the diagonals of [a]. On output
    {d},{e} and {f} are the diagonals of the decomposed matrix.

    x = LUsolve5(d,e,f,b).
    Solves [a]{x} = {b}, where {d}, {e} and {f} are the vectors
    returned from LUdecomp5.
    '''
def LUdecomp5(d,e,f):
    n = len(d)
    for k in range(n-2):
        lam = e[k]/d[k]
        d[k+1] = d[k+1] - lam*e[k]
        e[k+1] = e[k+1] - lam*f[k]
        e[k] = lam
        lam = f[k]/d[k]
        d[k+2] = d[k+2] - lam*f[k]
        f[k] = lam
    lam = e[n-2]/d[n-2]
    d[n-1] = d[n-1] - lam*e[n-2]
    e[n-2] = lam
    return d,e,f

def LUsolve5(d,e,f,b):
    n = len(d)
    b[1] = b[1] - e[0]*b[0]
    for k in range(2,n):
        b[k] = b[k] - e[k-1]*b[k-1] - f[k-2]*b[k-2]

    b[n-1] = b[n-1]/d[n-1]
    b[n-2] = b[n-2]/d[n-2] - e[n-2]*b[n-1]
    for k in range(n-3,-1,-1):
        b[k] = b[k]/d[k] - e[k]*b[k+1] - f[k]*b[k+2]
    return b






```


```python
#!/usr/bin/python
## example2_11
import numpy as np
from LUdecomp3 import *
d = np.ones((5))*2.0
c = np.ones((4))*(-1.0)
b = np.array([5.0, -5.0, 4.0, -5.0, 5.0])
e = c.copy()
c,d,e = LUdecomp3(c,d,e)
x = LUsolve3(c,d,e,b)
print("\nx =\n",x)
input("\nPress return to exit")
```


    x =
     [ 2. -1.  1. -1.  2.]

    Press return to exit





    ''



# 2.5 Pivoting

## Introduction

## Diagonal Dominance

## Gauss Elimination with Scaled Row Pivoting


```python
for k in range(0,n-1):

    # Find row containing element with largest relative size
    p = argmax(abs(a[k:n,k])/s[k:n]) + k

    # If this element is very small, matrix is singular    
    if abs(a[p,k]) < tol: error.err('Matrix is singular')

    # Check whether rows k and p must be interchanged
    if p != k:
        # Interchange rows if needed
        swap.swapRows(b,k,p)
        swap.swapRows(s,k,p)
        swap.swapRows(a,k,p)
    # Proceed with elimination
```

**swap**


```python
# %load code/swap.py
## module swap
''' swapRows(v,i,j).
    Swaps rows i and j of a vector or matrix [v].

    swapCols(v,i,j).
    Swaps columns of matrix [v].
'''
def swapRows(v,i,j):
    if len(v.shape) == 1:
        v[i],v[j] = v[j],v[i]
    else:
        v[[i,j],:] = v[[j,i],:]

def swapCols(v,i,j):
    v[:,[i,j]] = v[:,[j,i]]

```

**gaussPivot**


```python
# %load code/gaussPivot.py
## module gaussPivot
''' x = gaussPivot(a,b,tol=1.0e-12).
    Solves [a]{x} = {b} by Gauss elimination with
    scaled row pivoting
'''    
import numpy as np
import swap
import error

def gaussPivot(a,b,tol=1.0e-12):
    n = len(b)

  # Set up scale factors
    s = np.zeros(n)
    for i in range(n):
        s[i] = max(np.abs(a[i,:]))

    for k in range(0,n-1):

      # Row interchange, if needed
        p = np.argmax(np.abs(a[k:n,k])/s[k:n]) + k
        if abs(a[p,k]) < tol: error.err('Matrix is singular')
        if p != k:
            swap.swapRows(b,k,p)
            swap.swapRows(s,k,p)
            swap.swapRows(a,k,p)

      # Elimination
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                b[i] = b[i] - lam*b[k]
    if abs(a[n-1,n-1]) < tol: error.err('Matrix is singular')

  # Back substitution
    b[n-1] = b[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    return b





```

**LUpivot**


```python
# %load code/LUpivot.py
## module LUpivot
''' a,seq = LUdecomp(a,tol=1.0e-9).
    LU decomposition of matrix [a] using scaled row pivoting.
    The returned matrix [a] = contains [U] in the upper
    triangle and the nondiagonal terms of [L] in the lower triangle.
    Note that [L][U] is a row-wise permutation of the original [a];
    the permutations are recorded in the vector {seq}.

    x = LUsolve(a,b,seq).
    Solves [L][U]{x} = {b}, where the matrix [a] = and the
    permutation vector {seq} are returned from LUdecomp.
'''
import numpy as np
import swap
import error

def LUdecomp(a,tol=1.0e-9):
    n = len(a)
    seq = np.array(range(n))

  # Set up scale factors
    s = np.zeros((n))
    for i in range(n):
        s[i] = max(abs(a[i,:]))        

    for k in range(0,n-1):

      # Row interchange, if needed
        p = np.argmax(np.abs(a[k:n,k])/s[k:n]) + k
        if abs(a[p,k]) <  tol: error.err('Matrix is singular')
        if p != k:
            swap.swapRows(s,k,p)
            swap.swapRows(a,k,p)
            swap.swapRows(seq,k,p)

      # Elimination            
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a[i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                a[i,k] = lam
    return a,seq

def LUsolve(a,b,seq):
    n = len(a)

  # Rearrange constant vector; store it in [x]
    x = b.copy()
    for i in range(n):
        x[i] = b[seq[i]]

  # Solution
    for k in range(1,n):
        x[k] = x[k] - np.dot(a[k,0:k],x[0:k])
    x[n-1] = x[n-1]/a[n-1,n-1]    
    for k in range(n-2,-1,-1):
       x[k] = (x[k] - np.dot(a[k,k+1:n],x[k+1:n]))/a[k,k]
    return x



```

### When to Pivot

## When to Pivot

# 2.6 Matrix Inversion


```python
#!/usr/bin/python
## example2_13
import numpy as np
from LUpivot import *

def matInv(a):
    n = len(a[0])
    aInv = np.identity(n)
    a,seq = LUdecomp(a)

    for i in range(n):
        aInv[:,i] = LUsolve(a,aInv[:,i],seq)
    return aInv

a = np.array([[ 0.6, -0.4, 1.0],\
              [-0.3, 0.2, 0.5],\
              [ 0.6, -1.0, 0.5]])

aOrig = a.copy() # Save original [a]
aInv = matInv(a) # Invert [a] (original [a] is destroyed)

print("\naInv =\n",aInv)
print("\nCheck: a*aInv =\n", np.dot(aOrig,aInv))
input("\nPress return to exit")
```


    aInv =
     [[ 1.66666667 -2.22222222 -1.11111111]
     [ 1.25       -0.83333333 -1.66666667]
     [ 0.5         1.          0.        ]]

    Check: a*aInv =
     [[  1.00000000e+00  -4.44089210e-16  -1.18423789e-16]
     [  0.00000000e+00   1.00000000e+00   5.92118946e-17]
     [  0.00000000e+00  -3.33066907e-16   1.00000000e+00]]

    Press return to exit





    ''




```python
#!/usr/bin/python
## example2_14
import numpy as np
from LUdecomp3 import *
n = 6
d = np.ones((n))*2.0
e = np.ones((n-1))*(-1.0)
c = e.copy()
d[n-1] = 5.0
aInv = np.identity(n)
c,d,e = LUdecomp3(c,d,e)

for i in range(n):
    aInv[:,i] = LUsolve3(c,d,e,aInv[:,i])
print("\nThe inverse matrix is:\n",aInv)

input("\nPress return to exit")
```


    The inverse matrix is:
     [[ 0.84  0.68  0.52  0.36  0.2   0.04]
     [ 0.68  1.36  1.04  0.72  0.4   0.08]
     [ 0.52  1.04  1.56  1.08  0.6   0.12]
     [ 0.36  0.72  1.08  1.44  0.8   0.16]
     [ 0.2   0.4   0.6   0.8   1.    0.2 ]
     [ 0.04  0.08  0.12  0.16  0.2   0.24]]

    Press return to exit





    ''



# 2.7 Iterative Methods

## Intrduction

## Gauss-Seidel Method


```python
# %load code/gaussSeidel.py
## module gaussSeidel
''' x,numIter,omega = gaussSeidel(iterEqs,x,tol = 1.0e-9)
    Gauss-Seidel method for solving [A]{x} = {b}.
    The matrix [A] should be sparse. User must supply the
    function iterEqs(x,omega) that returns the improved {x},
    given the current {x} ('omega' is the relaxation factor).
'''
import numpy as np
import math

def gaussSeidel(iterEqs,x,tol = 1.0e-9):
    omega = 1.0
    k = 10
    p = 1
    for i in range(1,501):
        xOld = x.copy()
        x = iterEqs(x,omega)
        dx = math.sqrt(np.dot(x-xOld,x-xOld))
        if dx < tol: return x,i,omega
      # Compute relaxation factor after k+p iterations
        if i == k: dx1 = dx
        if i == k + p:
            dx2 = dx
            omega = 2.0/(1.0 + math.sqrt(1.0 - (dx2/dx1)**(1.0/p)))
    print('Gauss-Seidel failed to converge')

```

## Conjugate Gradient Method


```python
# %load code/conjGrad.py
## module conjGrad
''' x, numIter = conjGrad(Av,x,b,tol=1.0e-9)
    Conjugate gradient method for solving [A]{x} = {b}.
    The matrix [A] should be sparse. User must supply
    the function Av(v) that returns the vector [A]{v}
    and the starting vector x.
'''    
import numpy as np
import math

def conjGrad(Av,x,b,tol=1.0e-9):
    n = len(b)
    r = b - Av(x)
    s = r.copy()
    for i in range(n):
        u = Av(s)
        alpha = np.dot(s,r)/np.dot(s,u)
        x = x + alpha*s
        r = b - Av(x)
        if(math.sqrt(np.dot(r,r))) < tol:
            break
        else:
            beta = -np.dot(r,u)/np.dot(s,u)
            s = r + beta*s
    return x,i




```

# 2.8 Other Methods


```python

```
