
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

- [5 Numerical Differentiation](#5-numerical-differentiation)
- [5.1 Introduction](#51-introduction)
- [5.2 Finite Difference Approximations](#52-finite-difference-approximations)
	- [First Central Difference Approximations](#first-central-difference-approximations)
	- [First Noncentral Finite Difference Approximations](#first-noncentral-finite-difference-approximations)
	- [Second Noncentral Finite Difference Approximations](#second-noncentral-finite-difference-approximations)
	- [Errors in Finite Difference Approximations](#errors-in-finite-difference-approximations)
- [5.3 Richardson Extrapolation](#53-richardson-extrapolation)
- [5.4 Derivatives by Interpolation](#54-derivatives-by-interpolation)
	- [Polynomial Interpolant](#polynomial-interpolant)
	- [Cubic Spline Interpolant](#cubic-spline-interpolant)
		- [example5_4](#example5_4)

<!-- tocstop -->

# 5 Numerical Differentiation

# 5.1 Introduction

# 5.2 Finite Difference Approximations

## First Central Difference Approximations

## First Noncentral Finite Difference Approximations

## Second Noncentral Finite Difference Approximations

## Errors in Finite Difference Approximations

# 5.3 Richardson Extrapolation

# 5.4 Derivatives by Interpolation

## Polynomial Interpolant

## Cubic Spline Interpolant

### example5_4


```python
#!/usr/bin/python
## example5_4
from cubicSpline import curvatures
from LUdecomp3 import *
import numpy as np
xData = np.array([1.5,1.9,2.1,2.4,2.6,3.1])
yData = np.array([1.0628,1.3961,1.5432,1.7349,1.8423, 2.0397])
print(curvatures(xData,yData))
input("Press return to exit")
```

    [ 0.         -0.4258431  -0.37744139 -0.38796663 -0.55400477  0.        ]
    Press return to exit





    ''




```python

```
