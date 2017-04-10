
# Chapter 3: Symbolic computing
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Chapter 3: Symbolic computing](#chapter-3-symbolic-computing)
  * [Numbers](#numbers)
    * [Rationals](#rationals)
    * [Functions](#functions)
    * [Expressions](#expressions)
  * [Simplification](#simplification)
  * [Expand](#expand)
  * [Factor](#factor)
    * [Together, apart, cancel](#together-apart-cancel)
    * [Substitutions](#substitutions)
  * [Numerical evaluation](#numerical-evaluation)
  * [Calculus](#calculus)
  * [Integrals](#integrals)
  * [Series](#series)
  * [Limits](#limits)
  * [Sums and products](#sums-and-products)
  * [Equations](#equations)
  * [Linear algebra](#linear-algebra)
  * [Versions](#versions)

<!-- tocstop -->


---

Robert Johansson

Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).

The source code listings can be downloaded from http://www.apress.com/9781484205549


```python
import sympy
```


```python
sympy.init_printing()
```


```python
from sympy import I, pi, oo
```


```python
x = sympy.Symbol("x")
```


```python
y = sympy.Symbol("y", real=True)
```


```python
y.is_real
```




    True




```python
x.is_real is None
```




    True




```python
sympy.Symbol("z", imaginary=True).is_real
```




    False




```python
x = sympy.Symbol("x")
```


```python
y = sympy.Symbol("y", positive=True)
```


```python
sympy.sqrt(x ** 2)
```




$$\sqrt{x^{2}}$$




```python
sympy.sqrt(y ** 2)
```




$$y$$




```python
n1, n2, n3 = sympy.Symbol("n"), sympy.Symbol("n", integer=True), sympy.Symbol("n", odd=True)
```


```python
sympy.cos(n1 * pi)
```




$$\cos{\left (\pi n \right )}$$




```python
sympy.cos(n2 * pi)
```




$$\left(-1\right)^{n}$$




```python
sympy.cos(n3 * pi)
```




$$-1$$




```python
a, b, c = sympy.symbols("a, b, c", negative=True)
```


```python
d, e, f = sympy.symbols("d, e, f", positive=True)
```

## Numbers


```python
i = sympy.Integer(19)
```


```python
"i = {} [type {}]".format(i, type(i))
```




    "i = 19 [type <class 'sympy.core.numbers.Integer'>]"




```python
i.is_Integer, i.is_real, i.is_odd
```




    (True, True, True)




```python
f = sympy.Float(2.3)
```


```python
"f = {} [type {}]".format(f, type(f))
```




    "f = 2.30000000000000 [type <class 'sympy.core.numbers.Float'>]"




```python
f.is_Integer, f.is_real, f.is_odd
```




    (False, True, False)




```python
i, f = sympy.sympify(19), sympy.sympify(2.3)
```


```python
type(i)
```




    sympy.core.numbers.Integer




```python
type(f)
```




    sympy.core.numbers.Float




```python
n = sympy.Symbol("n", integer=True)
```


```python
n.is_integer, n.is_Integer, n.is_positive, n.is_Symbol
```




    (True, False, None, True)




```python
i = sympy.Integer(19)
```


```python
i.is_integer, i.is_Integer, i.is_positive, i.is_Symbol
```




    (True, True, True, False)




```python
i ** 50
```




$$8663234049605954426644038200675212212900743262211018069459689001$$




```python
sympy.factorial(100)
```




$$93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000$$




```python
"%.25f" % 0.3  # create a string represention with 25 decimals
```




    '0.2999999999999999888977698'




```python
sympy.Float(0.3, 25)
```




$$0.2999999999999999888977698$$




```python
sympy.Float('0.3', 25)
```




$$0.3$$



### Rationals


```python
sympy.Rational(11, 13)
```




$$\frac{11}{13}$$




```python
r1 = sympy.Rational(2, 3)
```


```python
r2 = sympy.Rational(4, 5)
```


```python
r1 * r2
```




$$\frac{8}{15}$$




```python
r1 / r2
```




$$\frac{5}{6}$$



### Functions


```python
x, y, z = sympy.symbols("x, y, z")
```


```python
f = sympy.Function("f")
```


```python
type(f)
```




    sympy.core.function.UndefinedFunction




```python
f(x)
```




$$f{\left (x \right )}$$




```python
g = sympy.Function("g")(x, y, z)
```


```python
g
```




$$g{\left (x,y,z \right )}$$




```python
g.free_symbols
```




$$\left\{x, y, z\right\}$$




```python
sympy.sin
```




    sin




```python
sympy.sin(x)
```




$$\sin{\left (x \right )}$$




```python
sympy.sin(pi * 1.5)
```




$$-1$$




```python
n = sympy.Symbol("n", integer=True)
```


```python
sympy.sin(pi * n)
```




$$0$$




```python
h = sympy.Lambda(x, x**2)
```


```python
h
```




$$\left( x \mapsto x^{2} \right)$$




```python
h(5)
```




$$25$$




```python
h(1+x)
```




$$\left(x + 1\right)^{2}$$



### Expressions


```python
x = sympy.Symbol("x")
```


```python
e = 1 + 2 * x**2 + 3 * x**3
```


```python
e
```




$$3 x^{3} + 2 x^{2} + 1$$




```python
e.args
```




$$\left ( 1, \quad 2 x^{2}, \quad 3 x^{3}\right )$$




```python
e.args[1]
```




$$2 x^{2}$$




```python
e.args[1].args[1]
```




$$x^{2}$$




```python
e.args[1].args[1].args[0]
```




$$x$$




```python
e.args[1].args[1].args[0].args
```




$$\left ( \right )$$



## Simplification


```python
expr = 2 * (x**2 - x) - x * (x + 1)
```


```python
expr
```




$$2 x^{2} - x \left(x + 1\right) - 2 x$$




```python
sympy.simplify(expr)
```




$$x \left(x - 3\right)$$




```python
expr.simplify()
```




$$x \left(x - 3\right)$$




```python
expr
```




$$2 x^{2} - x \left(x + 1\right) - 2 x$$




```python
expr = 2 * sympy.cos(x) * sympy.sin(x)
```


```python
expr
```




$$2 \sin{\left (x \right )} \cos{\left (x \right )}$$




```python
sympy.trigsimp(expr)
```




$$\sin{\left (2 x \right )}$$




```python
expr = sympy.exp(x) * sympy.exp(y)
```


```python
expr
```




$$e^{x} e^{y}$$




```python
sympy.powsimp(expr)
```




$$e^{x + y}$$



## Expand


```python
expr = (x + 1) * (x + 2)
```


```python
sympy.expand(expr)
```




$$x^{2} + 3 x + 2$$




```python
sympy.sin(x + y).expand(trig=True)
```




$$\sin{\left (x \right )} \cos{\left (y \right )} + \sin{\left (y \right )} \cos{\left (x \right )}$$




```python
a, b = sympy.symbols("a, b", positive=True)
```


```python
sympy.log(a * b).expand(log=True)
```




$$\log{\left (a \right )} + \log{\left (b \right )}$$




```python
sympy.exp(I*a + b).expand(complex=True)
```




$$i e^{b} \sin{\left (a \right )} + e^{b} \cos{\left (a \right )}$$




```python
sympy.expand((a * b)**x, power_exp=True)
```




$$a^{x} b^{x}$$




```python
sympy.exp(I*(a-b)*x).expand(power_exp=True)
```




$$e^{i a x} e^{- i b x}$$



## Factor


```python
sympy.factor(x**2 - 1)
```




$$\left(x - 1\right) \left(x + 1\right)$$




```python
sympy.factor(x * sympy.cos(y) + sympy.sin(z) * x)
```




$$x \left(\sin{\left (z \right )} + \cos{\left (y \right )}\right)$$




```python
sympy.logcombine(sympy.log(a) - sympy.log(b))
```




$$\log{\left (\frac{a}{b} \right )}$$




```python
expr = x + y + x * y * z
```


```python
expr.factor()
```




$$x y z + x + y$$




```python
expr.collect(x)
```




$$x \left(y z + 1\right) + y$$




```python
expr.collect(y)
```




$$x + y \left(x z + 1\right)$$




```python
expr = sympy.cos(x + y) + sympy.sin(x - y)
```


```python
expr.expand(trig=True).collect([sympy.cos(x), sympy.sin(x)]).collect(sympy.cos(y) - sympy.sin(y))
```




$$\left(\sin{\left (x \right )} + \cos{\left (x \right )}\right) \left(- \sin{\left (y \right )} + \cos{\left (y \right )}\right)$$



### Together, apart, cancel


```python
sympy.apart(1/(x**2 + 3*x + 2), x)
```




$$- \frac{1}{x + 2} + \frac{1}{x + 1}$$




```python
sympy.together(1 / (y * x + y) + 1 / (1+x))
```




$$\frac{y + 1}{y \left(x + 1\right)}$$




```python
sympy.cancel(y / (y * x + y))
```




$$\frac{1}{x + 1}$$



### Substitutions


```python
(x + y).subs(x, y)
```




$$2 y$$




```python
sympy.sin(x * sympy.exp(x)).subs(x, y)
```




$$\sin{\left (y e^{y} \right )}$$




```python
sympy.sin(x * z).subs({z: sympy.exp(y), x: y, sympy.sin: sympy.cos})
```




$$\cos{\left (y e^{y} \right )}$$




```python
expr = x * y + z**2 *x
```


```python
values = {x: 1.25, y: 0.4, z: 3.2}
```


```python
expr.subs(values)
```




$$13.3$$



## Numerical evaluation


```python
sympy.N(1 + pi)
```




$$4.14159265358979$$




```python
sympy.N(pi, 50)
```




$$3.1415926535897932384626433832795028841971693993751$$




```python
(x + 1/pi).evalf(7)
```




$$x + 0.3183099$$




```python
expr = sympy.sin(pi * x * sympy.exp(x))
```


```python
[expr.subs(x, xx).evalf(3) for xx in range(0, 10)]
```




$$\left [ 0, \quad 0.774, \quad 0.642, \quad 0.722, \quad 0.944, \quad 0.205, \quad 0.974, \quad 0.977, \quad -0.87, \quad -0.695\right ]$$




```python
expr_func = sympy.lambdify(x, expr)
```


```python
expr_func(1.0)
```




$$0.773942685266709$$




```python
expr_func = sympy.lambdify(x, expr, 'numpy')
```


```python
import numpy as np
```


```python
xvalues = np.arange(0, 10)
```


```python
expr_func(xvalues)
```




    array([ 0.        ,  0.77394269,  0.64198244,  0.72163867,  0.94361635,
            0.20523391,  0.97398794,  0.97734066, -0.87034418, -0.69512687])



## Calculus


```python
f = sympy.Function('f')(x)
```


```python
sympy.diff(f, x)
```




$$\frac{d}{d x} f{\left (x \right )}$$




```python
sympy.diff(f, x, x)
```




$$\frac{d^{2}}{d x^{2}}  f{\left (x \right )}$$




```python
sympy.diff(f, x, 3)
```




$$\frac{d^{3}}{d x^{3}}  f{\left (x \right )}$$




```python
g = sympy.Function('g')(x, y)
```


```python
g.diff(x, y)
```




$$\frac{\partial^{2}}{\partial x\partial y}  g{\left (x,y \right )}$$




```python
g.diff(x, 3, y, 2)         # equivalent to s.diff(g, x, x, x, y, y)
```




$$\frac{\partial^{5}}{\partial x^{3}\partial y^{2}}  g{\left (x,y \right )}$$




```python
expr = x**4 + x**3 + x**2 + x + 1
```


```python
expr.diff(x)
```




$$4 x^{3} + 3 x^{2} + 2 x + 1$$




```python
expr.diff(x, x)
```




$$2 \left(6 x^{2} + 3 x + 1\right)$$




```python
expr = (x + 1)**3 * y ** 2 * (z - 1)
```


```python
expr.diff(x, y, z)
```




$$6 y \left(x + 1\right)^{2}$$




```python
expr = sympy.sin(x * y) * sympy.cos(x / 2)
```


```python
expr.diff(x)
```




$$y \cos{\left (\frac{x}{2} \right )} \cos{\left (x y \right )} - \frac{1}{2} \sin{\left (\frac{x}{2} \right )} \sin{\left (x y \right )}$$




```python
expr = sympy.special.polynomials.hermite(x, 0)
```


```python
expr.diff(x).doit()
```




$$\frac{2^{x} \sqrt{\pi} \operatorname{polygamma}{\left (0,- \frac{x}{2} + \frac{1}{2} \right )}}{2 \Gamma{\left(- \frac{x}{2} + \frac{1}{2} \right)}} + \frac{2^{x} \sqrt{\pi} \log{\left (2 \right )}}{\Gamma{\left(- \frac{x}{2} + \frac{1}{2} \right)}}$$




```python
d = sympy.Derivative(sympy.exp(sympy.cos(x)), x)
```


```python
d
```




$$\frac{d}{d x} e^{\cos{\left (x \right )}}$$




```python
d.doit()
```




$$- e^{\cos{\left (x \right )}} \sin{\left (x \right )}$$



## Integrals


```python
a, b = sympy.symbols("a, b")
x, y = sympy.symbols('x, y')
f = sympy.Function('f')(x)
```


```python
sympy.integrate(f)
```




$$\int f{\left (x \right )}\, dx$$




```python
sympy.integrate(f, (x, a, b))
```




$$\int_{a}^{b} f{\left (x \right )}\, dx$$




```python
sympy.integrate(sympy.sin(x))
```




$$- \cos{\left (x \right )}$$




```python
sympy.integrate(sympy.sin(x), (x, a, b))
```




$$\cos{\left (a \right )} - \cos{\left (b \right )}$$




```python
sympy.integrate(sympy.exp(-x**2), (x, 0, oo))
```




$$\frac{\sqrt{\pi}}{2}$$




```python
a, b, c = sympy.symbols("a, b, c", positive=True)
```


```python
sympy.integrate(a * sympy.exp(-((x-b)/c)**2), (x, -oo, oo))
```




$$\sqrt{\pi} a c$$




```python
sympy.integrate(sympy.sin(x * sympy.cos(x)))
```




$$\int \sin{\left (x \cos{\left (x \right )} \right )}\, dx$$




```python
expr = sympy.sin(x*sympy.exp(y))
```


```python
sympy.integrate(expr, x)
```




$$- e^{- y} \cos{\left (x e^{y} \right )}$$




```python
expr = (x + y)**2
```


```python
sympy.integrate(expr, x)
```




$$\frac{x^{3}}{3} + x^{2} y + x y^{2}$$




```python
sympy.integrate(expr, x, y)
```




$$\frac{x^{3} y}{3} + \frac{x^{2} y^{2}}{2} + \frac{x y^{3}}{3}$$




```python
sympy.integrate(expr, (x, 0, 1), (y, 0, 1))
```




$$\frac{7}{6}$$



## Series


```python
x = sympy.Symbol("x")
```


```python
f = sympy.Function("f")(x)
```


```python
sympy.series(f, x)
```




$$f{\left (0 \right )} + x \left. \frac{d}{d x} f{\left (x \right )} \right|_{\substack{ x=0 }} + \frac{x^{2}}{2} \left. \frac{d^{2}}{d x^{2}}  f{\left (x \right )} \right|_{\substack{ x=0 }} + \frac{x^{3}}{6} \left. \frac{d^{3}}{d x^{3}}  f{\left (x \right )} \right|_{\substack{ x=0 }} + \frac{x^{4}}{24} \left. \frac{d^{4}}{d x^{4}}  f{\left (x \right )} \right|_{\substack{ x=0 }} + \frac{x^{5}}{120} \left. \frac{d^{5}}{d x^{5}}  f{\left (x \right )} \right|_{\substack{ x=0 }} + \mathcal{O}\left(x^{6}\right)$$




```python
x0 = sympy.Symbol("{x_0}")
```


```python
f.series(x, x0, n=2)
```




$$f{\left ({x_{0}} \right )} + \left(x - {x_{0}}\right) \left. \frac{d}{d \xi_{1}} f{\left (\xi_{1} \right )} \right|_{\substack{ \xi_{1}={x_{0}} }} + \mathcal{O}\left(\left(x - {x_{0}}\right)^{2}; x\rightarrow{x_{0}}\right)$$




```python
f.series(x, x0, n=2).removeO()
```




$$\left(x - {x_{0}}\right) \left. \frac{d}{d \xi_{1}} f{\left (\xi_{1} \right )} \right|_{\substack{ \xi_{1}={x_{0}} }} + f{\left ({x_{0}} \right )}$$




```python
sympy.cos(x).series()
```




$$1 - \frac{x^{2}}{2} + \frac{x^{4}}{24} + \mathcal{O}\left(x^{6}\right)$$




```python
sympy.sin(x).series()
```




$$x - \frac{x^{3}}{6} + \frac{x^{5}}{120} + \mathcal{O}\left(x^{6}\right)$$




```python
sympy.exp(x).series()
```




$$1 + x + \frac{x^{2}}{2} + \frac{x^{3}}{6} + \frac{x^{4}}{24} + \frac{x^{5}}{120} + \mathcal{O}\left(x^{6}\right)$$




```python
(1/(1+x)).series()
```




$$1 - x + x^{2} - x^{3} + x^{4} - x^{5} + \mathcal{O}\left(x^{6}\right)$$




```python
expr = sympy.cos(x) / (1 + sympy.sin(x * y))
```


```python
expr.series(x, n=4)
```




$$1 - x y + x^{2} \left(y^{2} - \frac{1}{2}\right) + x^{3} \left(- \frac{5 y^{3}}{6} + \frac{y}{2}\right) + \mathcal{O}\left(x^{4}\right)$$




```python
expr.series(y, n=4)
```




$$\cos{\left (x \right )} - x y \cos{\left (x \right )} + x^{2} y^{2} \cos{\left (x \right )} - \frac{5 x^{3}}{6} y^{3} \cos{\left (x \right )} + \mathcal{O}\left(y^{4}\right)$$




```python
expr.series(y).removeO().series(x).removeO().expand()
```




$$- \frac{61 x^{5}}{120} y^{5} + \frac{5 x^{5}}{12} y^{3} - \frac{x^{5} y}{24} + \frac{2 x^{4}}{3} y^{4} - \frac{x^{4} y^{2}}{2} + \frac{x^{4}}{24} - \frac{5 x^{3}}{6} y^{3} + \frac{x^{3} y}{2} + x^{2} y^{2} - \frac{x^{2}}{2} - x y + 1$$



## Limits


```python
sympy.limit(sympy.sin(x) / x, x, 0)
```




$$1$$




```python
f = sympy.Function('f')
x, h = sympy.symbols("x, h")
```


```python
diff_limit = (f(x + h) - f(x))/h
```


```python
sympy.limit(diff_limit.subs(f, sympy.cos), h, 0)
```




$$- \sin{\left (x \right )}$$




```python
sympy.limit(diff_limit.subs(f, sympy.sin), h, 0)
```




$$\cos{\left (x \right )}$$




```python
expr = (x**2 - 3*x) / (2*x - 2)
```


```python
p = sympy.limit(expr/x, x, oo)
```


```python
q = sympy.limit(expr - p*x, x, oo)
```


```python
p, q
```




$$\left ( \frac{1}{2}, \quad -1\right )$$



## Sums and products


```python
n = sympy.symbols("n", integer=True)
```


```python
x = sympy.Sum(1/(n**2), (n, 1, oo))
```


```python
x
```




$$\sum_{n=1}^{\infty} \frac{1}{n^{2}}$$




```python
x.doit()
```




$$\frac{\pi^{2}}{6}$$




```python
x = sympy.Product(n, (n, 1, 7))
```


```python
x
```




$$\prod_{n=1}^{7} n$$




```python
x.doit()
```




$$5040$$




```python
x = sympy.Symbol("x")
```


```python
sympy.Sum((x)**n/(sympy.factorial(n)), (n, 1, oo)).doit().simplify()
```




$$e^{x} - 1$$



## Equations


```python
x = sympy.symbols("x")
```


```python
sympy.solve(x**2 + 2*x - 3)
```




$$\left [ -3, \quad 1\right ]$$




```python
a, b, c = sympy.symbols("a, b, c")
```


```python
sympy.solve(a * x**2 + b * x + c, x)
```




$$\left [ \frac{1}{2 a} \left(- b + \sqrt{- 4 a c + b^{2}}\right), \quad - \frac{1}{2 a} \left(b + \sqrt{- 4 a c + b^{2}}\right)\right ]$$




```python
sympy.solve(sympy.sin(x) - sympy.cos(x), x)
```




$$\left [ - \frac{3 \pi}{4}, \quad \frac{\pi}{4}\right ]$$




```python
sympy.solve(sympy.exp(x) + 2 * x, x)
```




$$\left [ - \operatorname{LambertW}{\left (\frac{1}{2} \right )}\right ]$$




```python
sympy.solve(x**5 - x**2 + 1, x)
```




$$\left [ \operatorname{RootOf} {\left(x^{5} - x^{2} + 1, 0\right)}, \quad \operatorname{RootOf} {\left(x^{5} - x^{2} + 1, 1\right)}, \quad \operatorname{RootOf} {\left(x^{5} - x^{2} + 1, 2\right)}, \quad \operatorname{RootOf} {\left(x^{5} - x^{2} + 1, 3\right)}, \quad \operatorname{RootOf} {\left(x^{5} - x^{2} + 1, 4\right)}\right ]$$




```python
1 #s.solve(s.tan(x) - x, x)
```




$$1$$




```python
eq1 = x + 2 * y - 1
eq2 = x - y + 1
```


```python
sympy.solve([eq1, eq2], [x, y], dict=True)
```




$$\left [ \left \{ x : - \frac{1}{3}, \quad y : \frac{2}{3}\right \}\right ]$$




```python
eq1 = x**2 - y
eq2 = y**2 - x
```


```python
sols = sympy.solve([eq1, eq2], [x, y], dict=True)
```


```python
sols
```




$$\left [ \left \{ x : 0, \quad y : 0\right \}, \quad \left \{ x : 1, \quad y : 1\right \}, \quad \left \{ x : - \frac{1}{2} + \frac{\sqrt{3} i}{2}, \quad y : - \frac{1}{2} - \frac{\sqrt{3} i}{2}\right \}, \quad \left \{ x : \frac{1}{4} \left(1 - \sqrt{3} i\right)^{2}, \quad y : - \frac{1}{2} + \frac{\sqrt{3} i}{2}\right \}\right ]$$




```python
[eq1.subs(sol).simplify() == 0 and eq2.subs(sol).simplify() == 0 for sol in sols]
```




    [True, True, True, True]



## Linear algebra


```python
sympy.Matrix([1,2])
```




$$\left[\begin{matrix}1\\2\end{matrix}\right]$$




```python
sympy.Matrix([[1,2]])
```




$$\left[\begin{matrix}1 & 2\end{matrix}\right]$$




```python
sympy.Matrix([[1, 2], [3, 4]])
```




$$\left[\begin{matrix}1 & 2\\3 & 4\end{matrix}\right]$$




```python
sympy.Matrix(3, 4, lambda m,n: 10 * m + n)
```




$$\left[\begin{matrix}0 & 1 & 2 & 3\\10 & 11 & 12 & 13\\20 & 21 & 22 & 23\end{matrix}\right]$$




```python
a, b, c, d = sympy.symbols("a, b, c, d")
```


```python
M = sympy.Matrix([[a, b], [c, d]])
```


```python
M
```




$$\left[\begin{matrix}a & b\\c & d\end{matrix}\right]$$




```python
M * M
```




$$\left[\begin{matrix}a^{2} + b c & a b + b d\\a c + c d & b c + d^{2}\end{matrix}\right]$$




```python
x = sympy.Matrix(sympy.symbols("x_1, x_2"))
```


```python
M * x
```




$$\left[\begin{matrix}a x_{1} + b x_{2}\\c x_{1} + d x_{2}\end{matrix}\right]$$




```python
p, q = sympy.symbols("p, q")
```


```python
M = sympy.Matrix([[1, p], [q, 1]])
```


```python
M
```




$$\left[\begin{matrix}1 & p\\q & 1\end{matrix}\right]$$




```python
b = sympy.Matrix(sympy.symbols("b_1, b_2"))
```


```python
b
```




$$\left[\begin{matrix}b_{1}\\b_{2}\end{matrix}\right]$$




```python
x = M.solve(b)
x
```




$$\left[\begin{matrix}b_{1} \left(\frac{p q}{- p q + 1} + 1\right) - \frac{b_{2} p}{- p q + 1}\\- \frac{b_{1} q}{- p q + 1} + \frac{b_{2}}{- p q + 1}\end{matrix}\right]$$




```python
x = M.LUsolve(b)
```


```python
x
```




$$\left[\begin{matrix}b_{1} - \frac{p \left(- b_{1} q + b_{2}\right)}{- p q + 1}\\\frac{- b_{1} q + b_{2}}{- p q + 1}\end{matrix}\right]$$




```python
x = M.inv() * b
```


```python
x
```




$$\left[\begin{matrix}b_{1} \left(\frac{p q}{- p q + 1} + 1\right) - \frac{b_{2} p}{- p q + 1}\\- \frac{b_{1} q}{- p q + 1} + \frac{b_{2}}{- p q + 1}\end{matrix}\right]$$



## Versions


```python
%reload_ext version_information
%version_information sympy, numpy
```




<table><tr><th>Software</th><th>Version</th></tr><tr><td>Python</td><td>3.4.3 64bit [GCC 4.2.1 (Apple Inc. build 5577)]</td></tr><tr><td>IPython</td><td>3.2.0</td></tr><tr><td>OS</td><td>Darwin 14.5.0 x86_64 i386 64bit</td></tr><tr><td>sympy</td><td>0.7.6</td></tr><tr><td>numpy</td><td>1.9.2</td></tr></table>
