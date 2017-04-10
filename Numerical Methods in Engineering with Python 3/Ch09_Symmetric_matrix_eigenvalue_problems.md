
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

- [9 Symmetric Matrix Eigenvalue Problems](#9-symmetric-matrix-eigenvalue-problems)
- [9.1 Introduction](#91-introduction)
- [9.2 Jacobi Method](#92-jacobi-method)
	- [Similarity Transformation and Diagonalization](#similarity-transformation-and-diagonalization)
	- [Jacobi Rotation](#jacobi-rotation)
	- [Jacobi Diagonalization](#jacobi-diagonalization)
		- [jacobi](#jacobi)
		- [sortJacobi](#sortjacobi)
	- [Transformation to Standard Form](#transformation-to-standard-form)
		- [stdForm](#stdform)
		- [EXAMPLE9.3](#example93)
- [9.3 Power and Inverse Power Methods](#93-power-and-inverse-power-methods)
	- [Eigenvalue Shifting](#eigenvalue-shifting)
	- [Power Method](#power-method)
		- [inversePower](#inversepower)
		- [EXAMPLE9.4](#example94)
		- [EXAMPLE9.5](#example95)
	- [inversePower5](#inversepower5)
		- [EXAMPLE9.6](#example96)
- [9.4 Householder Reduction to Tridiagonal Form](#94-householder-reduction-to-tridiagonal-form)
	- [Householder Matrix](#householder-matrix)
	- [Householder Reduction of a Symmetric Matrix](#householder-reduction-of-a-symmetric-matrix)
	- [Accumulated Transformation Matrix](#accumulated-transformation-matrix)
		- [householder](#householder)
		- [EXAMPLE9.8](#example98)
- [9.5 Eigenvalues of Symmetric Tridiagonal Matrices](#95-eigenvalues-of-symmetric-tridiagonal-matrices)
	- [sturmSeq](#sturmseq)
		- [sturmSeq](#sturmseq-1)
	- [Gerschgorin’s Theorem](#gerschgorins-theorem)
		- [gerschgorin](#gerschgorin)
		- [lamRange](#lamrange)
	- [Computation of Eigenvalues](#computation-of-eigenvalues)
		- [eigenvals3](#eigenvals3)
		- [EXAMPLE9.12](#example912)
	- [Computation of Eigenvectors](#computation-of-eigenvectors)
		- [inversePower3](#inversepower3)
		- [EXAMPLE9.13](#example913)
		- [EXAMPLE9.14](#example914)
- [9.6 Other Methods](#96-other-methods)

<!-- tocstop -->


# 9 Symmetric Matrix Eigenvalue Problems

# 9.1 Introduction

# 9.2 Jacobi Method

## Similarity Transformation and Diagonalization

## Jacobi Rotation

## Jacobi Diagonalization

### jacobi


```python
# %load code/jacobi.py
## module jacobi
''' lam,x = jacobi(a,tol = 1.0e-8).
    Solution of std. eigenvalue problem [a]{x} = lam{x}
    by Jacobi's method. Returns eigenvalues in vector {lam}
    and the eigenvectors as columns of matrix [x].
'''
import numpy as np
import math

def jacobi(a,tol = 1.0e-8): # Jacobi method

    def threshold(a):
        sum = 0.0
        for i in range(n-1):
            for j in range (i+1,n):
                sum = sum + abs(a[i,j])
        return 0.5*sum/n/(n-1)

    def rotate(a,p,k,l): # Rotate to make a[k,l] = 0
        aDiff = a[l,l] - a[k,k]
        if abs(a[k,l]) < abs(aDiff)*1.0e-36: t = a[k,l]/aDiff
        else:
            phi = aDiff/(2.0*a[k,l])
            t = 1.0/(abs(phi) + math.sqrt(phi**2 + 1.0))
            if phi < 0.0: t = -t
        c = 1.0/math.sqrt(t**2 + 1.0); s = t*c
        tau = s/(1.0 + c)
        temp = a[k,l]
        a[k,l] = 0.0
        a[k,k] = a[k,k] - t*temp
        a[l,l] = a[l,l] + t*temp
        for i in range(k):      # Case of i < k
            temp = a[i,k]
            a[i,k] = temp - s*(a[i,l] + tau*temp)
            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
        for i in range(k+1,l):  # Case of k < i < l
            temp = a[k,i]
            a[k,i] = temp - s*(a[i,l] + tau*a[k,i])
            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
        for i in range(l+1,n):  # Case of i > l
            temp = a[k,i]
            a[k,i] = temp - s*(a[l,i] + tau*temp)
            a[l,i] = a[l,i] + s*(temp - tau*a[l,i])
        for i in range(n):      # Update transformation matrix
            temp = p[i,k]
            p[i,k] = temp - s*(p[i,l] + tau*p[i,k])
            p[i,l] = p[i,l] + s*(temp - tau*p[i,l])

    n = len(a)        
    p = np.identity(n,float)
    for k in range(20):
        mu = threshold(a)       # Compute new threshold
        for i in range(n-1):    # Sweep through matrix
            for j in range(i+1,n):   
                if abs(a[i,j]) >= mu:
                    rotate(a,p,i,j)
        if mu <= tol: return np.diagonal(a),p
    print('Jacobi method did not converge')


```

### sortJacobi


```python
# %load code/sortJacobi.py
## module sortJacobi
''' sortJacobi(lam,x).
    Sorts the eigenvalues {lam} and eigenvectors [x]
    in order of ascending eigenvalues.
'''    
import swap

def sortJacobi(lam,x):
    n = len(lam)
    for i in range(n-1):
        index = i
        val = lam[i]
        for j in range(i+1,n):
            if lam[j] < val:
                index = j
                val = lam[j]
        if index != i:
            swap.swapRows(lam,i,index)
            swap.swapCols(x,i,index)

```

## Transformation to Standard Form

### stdForm


```python
# %load code/stdForm.py
## module stdForm
''' h,t = stdForm(a,b).
    Transforms the eigenvalue problem [a]{x} = lam[b]{x}
    to the standard form [h]{z} = lam{z}. The eigenvectors
    are related by {x} = [t]{z}.
'''    
import numpy as np
from choleski import *

def stdForm(a,b):

    def invert(L): # Inverts lower triangular matrix L
        n = len(L)
        for j in range(n-1):
            L[j,j] = 1.0/L[j,j]
            for i in range(j+1,n):
                L[i,j] = -np.dot(L[i,j:i],L[j:i,j])/L[i,i]
        L[n-1,n-1] = 1.0/L[n-1,n-1]

    n = len(a)
    L = choleski(b)          
    invert(L)
    h = np.dot(b,np.inner(a,L))
    return h,np.transpose(L)







```

### EXAMPLE9.3


```python
#!/usr/bin/python
## example9_3
import numpy
from jacobi import *
import math
from sortJacobi import *
from stdForm import *

A = np.array([[ 1/3, -1/3, 0.0], \
              [-1/3, 4/3, -1.0], \
              [ 0.0, -1.0, 2.0]])
B = np.array([[1.0, 0.0, 0.0], \
              [0.0, 1.0, 0.0], \
              [0.0, 0.0, 2.0]])

H,T = stdForm(A,B) # Transform into std. form
lam,Z = jacobi(H) # Z = eigenvecs. of H
X = np.dot(T,Z) # Eigenvecs. of original problem
sortJacobi(lam,X) # Arrange in ascending order of eigenvecs.

for i in range(3): # Normalize eigenvecs.
    X[:,i] = X[:,i]/math.sqrt(np.dot(X[:,i],X[:,i]))

print('Eigenvalues:\n',lam)
print('Eigenvectors:\n',X)
input ("Press return to exit")
```

# 9.3 Power and Inverse Power Methods

## Eigenvalue Shifting

## Power Method

### inversePower


```python
# %load code/inversePower.py
## module inversePower
''' lam,x = inversePower(a,s,tol=1.0e-6).
    Inverse power method for solving the eigenvalue problem
    [a]{x} = lam{x}. Returns 'lam' closest to 's' and the
    corresponding eigenvector {x}.
'''
import numpy as np
from LUdecomp import *
import math
from random import random
def inversePower(a,s,tol=1.0e-6):
    n = len(a)
    aStar = a - np.identity(n)*s  # Form [a*] = [a] - s[I]
    aStar = LUdecomp(aStar)       # Decompose [a*]
    x = np.zeros(n)
    for i in range(n):            # Seed [x] with random numbers
        x[i] = random()
    xMag = math.sqrt(np.dot(x,x)) # Normalize [x]
    x =x/xMag
    for i in range(50):           # Begin iterations      
        xOld = x.copy()           # Save current [x]
        x = LUsolve(aStar,x)      # Solve [a*][x] = [xOld]
        xMag = math.sqrt(np.dot(x,x)) # Normalize [x]
        x = x/xMag
        if np.dot(xOld,x) < 0.0:  # Detect change in sign of [x]
            sign = -1.0
            x = -x
        else: sign = 1.0
        if math.sqrt(np.dot(xOld - x,xOld - x)) < tol:
            return s + sign/xMag,x
    print('Inverse power method did not converge')

```

### EXAMPLE9.4


```python
#!/usr/bin/python
## example9_4
import numpy as np
import math

s = np.array([[-30.0, 10.0, 20.0], \
              [ 10.0, 40.0, -50.0], \
              [ 20.0, -50.0, -10.0]])

v = np.array([1.0, 0.0, 0.0])

for i in range(100):
    vOld = v.copy()
    z = np.dot(s,v)
    zMag = math.sqrt(np.dot(z,z))
    v = z/zMag

    if np.dot(vOld,v) < 0.0:
        sign = -1.0
        v = -v
    else: sign = 1.0

    if math.sqrt(np.dot(vOld - v,vOld - v)) < 1.0e-6: break

lam = sign*zMag
print("Number of iterations =",i)
print("Eigenvalue =",lam)
input("Press return to exit")
```

    Number of iterations = 92
    Eigenvalue = 70.94348330679053
    Press return to exit





    ''



### EXAMPLE9.5


```python
#!/usr/bin/python
## example9_5
import numpy as np
from inversePower import *

s = 5.0
a = np.array([[ 11.0, 2.0, 3.0, 1.0, 4.0], \
              [ 2.0, 9.0, 3.0, 5.0, 2.0], \
              [ 3.0, 3.0, 15.0, 4.0, 3.0], \
              [ 1.0, 5.0, 4.0, 12.0, 4.0], \
              [ 4.0, 2.0, 3.0, 4.0, 17.0]])

lam,x = inversePower(a,s)
print("Eigenvalue =",lam)
print("\nEigenvector:\n",x)
input("\nPrint press return to exit")
```

    Eigenvalue = 4.8739463786491815

    Eigenvector:
     [ 0.26726605 -0.74142853 -0.05017272  0.59491453 -0.14970634]

    Print press return to exit





    ''



## inversePower5


```python
#!/usr/bin/python
## example9_5
import numpy as np
from inversePower import *

s = 5.0
a = np.array([[ 11.0, 2.0, 3.0, 1.0, 4.0], \
              [ 2.0, 9.0, 3.0, 5.0, 2.0], \
              [ 3.0, 3.0, 15.0, 4.0, 3.0], \
              [ 1.0, 5.0, 4.0, 12.0, 4.0], \
              [ 4.0, 2.0, 3.0, 4.0, 17.0]])

lam,x = inversePower(a,s)
print("Eigenvalue =",lam)
print("\nEigenvector:\n",x)
input("\nPrint press return to exit")
```

    Eigenvalue = 4.873946378649211

    Eigenvector:
     [-0.26726603  0.74142854  0.05017272 -0.59491453  0.14970633]

    Print press return to exit





    ''



### EXAMPLE9.6


```python
#!/usr/bin/python
## example9_6
import numpy as np
from inversePower5 import *

def Bv(v): # Compute {z} = [B]{v}
    n = len(v)
    z = np.zeros(n)
    z[0] = 2.0*v[0] - v[1]

    for i in range(1,n-1):
        z[i] = -v[i-1] + 2.0*v[i] - v[i+1]
    z[n-1] = -v[n-2] + 2.0*v[n-1]
    return z

n = 100 # Number of interior nodes
d = np.ones(n)*6.0 # Specify diagonals of [A] = [f\e\d\e\f]
d[0] = 5.0
d[n-1] = 7.0
e = np.ones(n-1)*(-4.0)
f = np.ones(n-2)*1.0
lam,x = inversePower5(Bv,d,e,f)
print("PL^2/EI =",lam*(n+1)**2)
input("\nPress return to exit")
```

    PL^2/EI = 20.18673210142833

    Press return to exit





    ''



# 9.4 Householder Reduction to Tridiagonal Form

## Householder Matrix

## Householder Reduction of a Symmetric Matrix

## Accumulated Transformation Matrix

### householder


```python
# %load code/householder.py
## module householder
''' d,c = householder(a).
    Householder similarity transformation of matrix [a] to
    tridiagonal form].

    p = computeP(a).
    Computes the acccumulated transformation matrix [p]
    after calling householder(a).
'''    
import numpy as np
import math

def householder(a):
    n = len(a)
    for k in range(n-2):
        u = a[k+1:n,k]
        uMag = math.sqrt(np.dot(u,u))
        if u[0] < 0.0: uMag = -uMag
        u[0] = u[0] + uMag
        h = np.dot(u,u)/2.0
        v = np.dot(a[k+1:n,k+1:n],u)/h
        g = np.dot(u,v)/(2.0*h)
        v = v - g*u
        a[k+1:n,k+1:n] = a[k+1:n,k+1:n] - np.outer(v,u) \
                         - np.outer(u,v)
        a[k,k+1] = -uMag
    return np.diagonal(a),np.diagonal(a,1)

def computeP(a):
    n = len(a)
    p = np.identity(n)*1.0
    for k in range(n-2):
        u = a[k+1:n,k]
        h = np.dot(u,u)/2.0
        v = np.dot(p[1:n,k+1:n],u)/h           
        p[1:n,k+1:n] = p[1:n,k+1:n] - np.outer(v,u)
    return p




```

### EXAMPLE9.8


```python
#!/usr/bin/python
## example9_8
import numpy as np
from householder import *
a = np.array([[ 7.0, 2.0, 3.0, -1.0], \
              [ 2.0, 8.0, 5.0, 1.0], \
              [ 3.0, 5.0, 12.0, 9.0], \
              [-1.0, 1.0, 9.0, 7.0]])

d,c = householder(a)
print("Principal diagonal {d}:\n", d)
print("\nSubdiagonal {c}:\n",c)
print("\nTransformation matrix [P]:")
print(computeP(a))
input("\nPress return to exit")
```

# 9.5 Eigenvalues of Symmetric Tridiagonal Matrices

## sturmSeq
### sturmSeq


```python
# %load code/sturmSeq.py
## module sturmSeq
''' p = sturmSeq(c,d,lam).
    Returns the Sturm sequence {p[0],p[1],...,p[n]}
    associated with the characteristic polynomial
    |[A] - lam[I]| = 0, where [A] is a n x n
    tridiagonal matrix.

    numLam = numLambdas(p).
    Returns the number of eigenvalues of a tridiagonal
    matrix that are smaller than 'lam'.
    Uses the Sturm sequence {p} obtained from 'sturmSeq'.
'''
import numpy as np

def sturmSeq(d,c,lam):
    n = len(d) + 1
    p = np.ones(n)
    p[1] = d[0] - lam
    for i in range(2,n):
        p[i] = (d[i-1] - lam)*p[i-1] - (c[i-2]**2)*p[i-2]
    return p

def numLambdas(p):
    n = len(p)
    signOld = 1
    numLam = 0
    for i in range(1,n):
        if p[i] > 0.0: sign = 1
        elif p[i] < 0.0: sign = -1
        else: sign = -signOld
        if sign*signOld < 0: numLam = numLam + 1
        signOld = sign
    return numLam

```

## Gerschgorin’s Theorem

### gerschgorin


```python
# %load code/gerschgorin.py
## module gerschgorin
''' lamMin,lamMax = gerschgorin(d,c).
    Applies Gerschgorin's theorem to find the global bounds on
    the eigenvalues of a symmetric tridiagonal matrix.
'''
def gerschgorin(d,c):
    n = len(d)
    lamMin = d[0] - abs(c[0])
    lamMax = d[0] + abs(c[0])
    for i in range(1,n-1):
        lam = d[i] - abs(c[i]) - abs(c[i-1])
        if lam < lamMin: lamMin = lam
        lam = d[i] + abs(c[i]) + abs(c[i-1])
        if lam > lamMax: lamMax = lam
    lam = d[n-1] - abs(c[n-2])
    if lam < lamMin: lamMin = lam
    lam = d[n-1] + abs(c[n-2])
    if lam > lamMax: lamMax = lam
    return lamMin,lamMax

```

### lamRange


```python
# %load code/lamRange.py
## module lamRange
''' r = lamRange(d,c,N).
    Returns the sequence {r[0],r[1],...,r[N]} that
    separates the N lowest eigenvalues of the tridiagonal
    matrix; that is, r[i] < lam[i] < r[i+1].
'''
import numpy as np
from sturmSeq import *
from gerschgorin import *

def lamRange(d,c,N):
    lamMin,lamMax = gerschgorin(d,c)
    r = np.ones(N+1)
    r[0] = lamMin
  # Search for eigenvalues in descending order  
    for k in range(N,0,-1):
      # First bisection of interval(lamMin,lamMax)
        lam = (lamMax + lamMin)/2.0
        h = (lamMax - lamMin)/2.0
        for i in range(1000):
          # Find number of eigenvalues less than lam
            p = sturmSeq(d,c,lam)
            numLam = numLambdas(p)
          # Bisect again & find the half containing lam
            h = h/2.0
            if numLam < k: lam = lam + h
            elif numLam > k: lam = lam - h
            else: break
      # If eigenvalue located, change the upper limit
      # of search and record it in [r]
        lamMax = lam
        r[k] = lam
    return r

```

## Computation of Eigenvalues

### eigenvals3


```python
# %load code/eigenvals3.py
## module eigenvals3
''' lam = eigenvals3(d,c,N).
    Returns the N smallest eigenvalues of a symmetric
    tridiagonal matrix defined by its diagonals d and c.
'''    
from lamRange import *
from ridder import *
from sturmSeq import sturmSeq
from numpy import zeros

def eigenvals3(d,c,N):

    def f(x):             # f(x) = |[A] - x[I]|
        p = sturmSeq(d,c,x)
        return p[len(p)-1]

    lam = zeros(N)
    r = lamRange(d,c,N)   # Bracket eigenvalues
    for i in range(N):    # Solve by Brent's method
        lam[i] = ridder(f,r[i],r[i+1])
    return lam   


```

### EXAMPLE9.12


```python
#!/usr/bin/python
## example9_12
import numpy as np
from eigenvals3 import *
N = 3
n = 100
d = np.ones(n)*2.0
c = np.ones(n-1)*(-1.0)

lambdas = eigenvals3(d,c,N)

print(lambdas)
input("\nPress return to exit")
```

    [ 0.00096744  0.00386881  0.0087013 ]

    Press return to exit





    ''



## Computation of Eigenvectors

### inversePower3


```python
# %load code/inversePower3.py
## module inversePower3
''' lam,x = inversePower3(d,c,s,tol=1.0e-6)
    Inverse power method applied to a symmetric tridiagonal
    matrix. Returns the eigenvalue closest to s
    and the corresponding eigenvector.
'''
from LUdecomp3 import *
import math
import numpy as np
from numpy.random import rand

def inversePower3(d,c,s,tol=1.0e-6):
    n = len(d)
    e = c.copy()
    cc = c.copy()
    dStar = d - s                  # Form [A*] = [A] - s[I]
    LUdecomp3(cc,dStar,e)          # Decompose [A*]
    x = rand(n)                    # Seed x with random numbers
    xMag = math.sqrt(np.dot(x,x))  # Normalize [x]
    x = x/xMag

    for i in range(30):               # Begin iterations    
        xOld = x.copy()               # Save current [x]
        LUsolve3(cc,dStar,e,x)        # Solve [A*][x] = [xOld]
        xMag = math.sqrt(np.dot(x,x)) # Normalize [x]
        x = x/xMag
        if np.dot(xOld,x) < 0.0:   # Detect change in sign of [x]
            sign = -1.0
            x = -x
        else: sign = 1.0
        if math.sqrt(np.dot(xOld - x,xOld - x)) < tol:
            return s + sign/xMag,x
    print('Inverse power method did not converge')
    return

```

### EXAMPLE9.13


```python
#!/usr/bin/python
## example9_13
import numpy as np
from lamRange import *
from inversePower3 import *
N = 10
n = 100
d = np.ones(n)*2.0
c = np.ones(n-1)*(-1.0)
r = lamRange(d,c,N) # Bracket N smallest eigenvalues
s = (r[N-1] + r[N])/2.0 # Shift to midpoint of Nth bracket
lam,x = inversePower3(d,c,s) # Inverse power method
print("Eigenvalue No.",N," =",lam)
input("\nPress return to exit")
```

    Eigenvalue No. 10  = 0.0959737849345

    Press return to exit





    ''



### EXAMPLE9.14


```python
#!/usr/bin/python
## example9_14
from householder import *
from eigenvals3 import *
from inversePower3 import *
import numpy as np
N = 3 # Number of eigenvalues requested

a = np.array([[ 11.0, 2.0, 3.0, 1.0, 4.0], \
              [ 2.0, 9.0, 3.0, 5.0, 2.0], \
              [ 3.0, 3.0, 15.0, 4.0, 3.0], \
              [ 1.0, 5.0, 4.0, 12.0, 4.0], \
              [ 4.0, 2.0, 3.0, 4.0, 17.0]])

xx = np.zeros((len(a),N))
d,c = householder(a) # Tridiagonalize [A]
p = computeP(a) # Compute transformation matrix

lambdas = eigenvals3(d,c,N) # Compute eigenvalues

for i in range(N):
    s = lambdas[i]*1.0000001 # Shift very close to eigenvalue
    lam,x = inversePower3(d,c,s) # Compute eigenvector [x]
    xx[:,i] = x # Place [x] in array [xx]

xx = np.dot(p,xx) # Recover eigenvectors of [A]
print("Eigenvalues:\n",lambdas)
print("\nEigenvectors:\n",xx)
input("Press return to exit")
```

    Eigenvalues:
     [  4.87394638   8.66356791  10.93677451]

    Eigenvectors:
     [[ 0.26726603  0.72910002  0.50579164]
     [-0.74142854  0.41391448 -0.31882387]
     [-0.05017271 -0.4298639   0.52077788]
     [ 0.59491453  0.06955611 -0.60290543]
     [-0.14970633 -0.32782151 -0.08843985]]
    Press return to exit





    ''



# 9.6 Other Methods
