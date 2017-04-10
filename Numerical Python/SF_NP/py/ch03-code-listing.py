
# coding: utf-8

# # Chapter 3: Symbolic computing

# Robert Johansson
# 
# Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).
# 
# The source code listings can be downloaded from http://www.apress.com/9781484205549

# In[1]:

import sympy


# In[2]:

sympy.init_printing()


# In[3]:

from sympy import I, pi, oo


# In[4]:

x = sympy.Symbol("x")


# In[5]:

y = sympy.Symbol("y", real=True)


# In[6]:

y.is_real


# In[7]:

x.is_real is None


# In[8]:

sympy.Symbol("z", imaginary=True).is_real


# In[9]:

x = sympy.Symbol("x")


# In[10]:

y = sympy.Symbol("y", positive=True)


# In[11]:

sympy.sqrt(x ** 2)


# In[12]:

sympy.sqrt(y ** 2)


# In[13]:

n1, n2, n3 = sympy.Symbol("n"), sympy.Symbol("n", integer=True), sympy.Symbol("n", odd=True)


# In[14]:

sympy.cos(n1 * pi)


# In[15]:

sympy.cos(n2 * pi)


# In[16]:

sympy.cos(n3 * pi)


# In[17]:

a, b, c = sympy.symbols("a, b, c", negative=True)


# In[18]:

d, e, f = sympy.symbols("d, e, f", positive=True)


# ## Numbers

# In[19]:

i = sympy.Integer(19)


# In[20]:

"i = {} [type {}]".format(i, type(i))


# In[21]:

i.is_Integer, i.is_real, i.is_odd


# In[22]:

f = sympy.Float(2.3)


# In[23]:

"f = {} [type {}]".format(f, type(f))


# In[24]:

f.is_Integer, f.is_real, f.is_odd


# In[25]:

i, f = sympy.sympify(19), sympy.sympify(2.3)


# In[26]:

type(i)


# In[27]:

type(f)


# In[28]:

n = sympy.Symbol("n", integer=True)


# In[29]:

n.is_integer, n.is_Integer, n.is_positive, n.is_Symbol


# In[30]:

i = sympy.Integer(19)


# In[31]:

i.is_integer, i.is_Integer, i.is_positive, i.is_Symbol


# In[32]:

i ** 50


# In[33]:

sympy.factorial(100)


# In[34]:

"%.25f" % 0.3  # create a string represention with 25 decimals


# In[35]:

sympy.Float(0.3, 25)


# In[36]:

sympy.Float('0.3', 25)


# ### Rationals

# In[37]:

sympy.Rational(11, 13)


# In[38]:

r1 = sympy.Rational(2, 3)


# In[39]:

r2 = sympy.Rational(4, 5)


# In[40]:

r1 * r2


# In[41]:

r1 / r2


# ### Functions

# In[42]:

x, y, z = sympy.symbols("x, y, z")


# In[43]:

f = sympy.Function("f")


# In[44]:

type(f)


# In[45]:

f(x)


# In[46]:

g = sympy.Function("g")(x, y, z)


# In[47]:

g


# In[48]:

g.free_symbols


# In[49]:

sympy.sin


# In[50]:

sympy.sin(x)


# In[51]:

sympy.sin(pi * 1.5)


# In[52]:

n = sympy.Symbol("n", integer=True)


# In[53]:

sympy.sin(pi * n)


# In[54]:

h = sympy.Lambda(x, x**2)


# In[55]:

h


# In[56]:

h(5)


# In[57]:

h(1+x)


# ### Expressions

# In[58]:

x = sympy.Symbol("x")


# In[59]:

e = 1 + 2 * x**2 + 3 * x**3


# In[60]:

e


# In[61]:

e.args


# In[62]:

e.args[1]


# In[63]:

e.args[1].args[1]


# In[64]:

e.args[1].args[1].args[0]


# In[65]:

e.args[1].args[1].args[0].args


# ## Simplification

# In[66]:

expr = 2 * (x**2 - x) - x * (x + 1)


# In[67]:

expr


# In[68]:

sympy.simplify(expr)


# In[69]:

expr.simplify()


# In[70]:

expr


# In[71]:

expr = 2 * sympy.cos(x) * sympy.sin(x)


# In[72]:

expr


# In[73]:

sympy.trigsimp(expr)


# In[74]:

expr = sympy.exp(x) * sympy.exp(y)


# In[75]:

expr


# In[76]:

sympy.powsimp(expr)


# ## Expand

# In[77]:

expr = (x + 1) * (x + 2)


# In[78]:

sympy.expand(expr)


# In[79]:

sympy.sin(x + y).expand(trig=True)


# In[80]:

a, b = sympy.symbols("a, b", positive=True)


# In[81]:

sympy.log(a * b).expand(log=True)


# In[82]:

sympy.exp(I*a + b).expand(complex=True)


# In[83]:

sympy.expand((a * b)**x, power_exp=True)


# In[84]:

sympy.exp(I*(a-b)*x).expand(power_exp=True)


# ## Factor

# In[85]:

sympy.factor(x**2 - 1)


# In[86]:

sympy.factor(x * sympy.cos(y) + sympy.sin(z) * x)


# In[87]:

sympy.logcombine(sympy.log(a) - sympy.log(b))


# In[88]:

expr = x + y + x * y * z


# In[89]:

expr.factor()


# In[90]:

expr.collect(x)


# In[91]:

expr.collect(y)


# In[92]:

expr = sympy.cos(x + y) + sympy.sin(x - y)


# In[93]:

expr.expand(trig=True).collect([sympy.cos(x), sympy.sin(x)]).collect(sympy.cos(y) - sympy.sin(y))


# ### Together, apart, cancel

# In[94]:

sympy.apart(1/(x**2 + 3*x + 2), x)


# In[95]:

sympy.together(1 / (y * x + y) + 1 / (1+x))


# In[96]:

sympy.cancel(y / (y * x + y))


# ### Substitutions

# In[97]:

(x + y).subs(x, y)


# In[98]:

sympy.sin(x * sympy.exp(x)).subs(x, y)


# In[99]:

sympy.sin(x * z).subs({z: sympy.exp(y), x: y, sympy.sin: sympy.cos})


# In[100]:

expr = x * y + z**2 *x


# In[101]:

values = {x: 1.25, y: 0.4, z: 3.2}


# In[102]:

expr.subs(values)


# ## Numerical evaluation

# In[103]:

sympy.N(1 + pi)


# In[104]:

sympy.N(pi, 50)


# In[105]:

(x + 1/pi).evalf(7)


# In[106]:

expr = sympy.sin(pi * x * sympy.exp(x))


# In[107]:

[expr.subs(x, xx).evalf(3) for xx in range(0, 10)]


# In[108]:

expr_func = sympy.lambdify(x, expr)


# In[109]:

expr_func(1.0)


# In[110]:

expr_func = sympy.lambdify(x, expr, 'numpy')


# In[111]:

import numpy as np


# In[112]:

xvalues = np.arange(0, 10)


# In[113]:

expr_func(xvalues)


# ## Calculus

# In[114]:

f = sympy.Function('f')(x)


# In[115]:

sympy.diff(f, x)


# In[116]:

sympy.diff(f, x, x)


# In[117]:

sympy.diff(f, x, 3)


# In[118]:

g = sympy.Function('g')(x, y)


# In[119]:

g.diff(x, y)


# In[120]:

g.diff(x, 3, y, 2)         # equivalent to s.diff(g, x, x, x, y, y)


# In[121]:

expr = x**4 + x**3 + x**2 + x + 1


# In[122]:

expr.diff(x)


# In[123]:

expr.diff(x, x)


# In[124]:

expr = (x + 1)**3 * y ** 2 * (z - 1)


# In[125]:

expr.diff(x, y, z)


# In[126]:

expr = sympy.sin(x * y) * sympy.cos(x / 2)


# In[127]:

expr.diff(x)


# In[128]:

expr = sympy.special.polynomials.hermite(x, 0)


# In[129]:

expr.diff(x).doit()


# In[130]:

d = sympy.Derivative(sympy.exp(sympy.cos(x)), x)


# In[131]:

d


# In[132]:

d.doit()


# ## Integrals

# In[133]:

a, b = sympy.symbols("a, b")
x, y = sympy.symbols('x, y')
f = sympy.Function('f')(x)


# In[134]:

sympy.integrate(f)


# In[135]:

sympy.integrate(f, (x, a, b))


# In[136]:

sympy.integrate(sympy.sin(x))


# In[137]:

sympy.integrate(sympy.sin(x), (x, a, b))


# In[138]:

sympy.integrate(sympy.exp(-x**2), (x, 0, oo))


# In[139]:

a, b, c = sympy.symbols("a, b, c", positive=True)


# In[140]:

sympy.integrate(a * sympy.exp(-((x-b)/c)**2), (x, -oo, oo))


# In[141]:

sympy.integrate(sympy.sin(x * sympy.cos(x)))


# In[142]:

expr = sympy.sin(x*sympy.exp(y))


# In[143]:

sympy.integrate(expr, x)


# In[144]:

expr = (x + y)**2


# In[145]:

sympy.integrate(expr, x)


# In[146]:

sympy.integrate(expr, x, y)


# In[147]:

sympy.integrate(expr, (x, 0, 1), (y, 0, 1))


# ## Series

# In[148]:

x = sympy.Symbol("x")


# In[149]:

f = sympy.Function("f")(x)


# In[150]:

sympy.series(f, x)


# In[151]:

x0 = sympy.Symbol("{x_0}")


# In[152]:

f.series(x, x0, n=2)


# In[153]:

f.series(x, x0, n=2).removeO()


# In[154]:

sympy.cos(x).series()


# In[155]:

sympy.sin(x).series()


# In[156]:

sympy.exp(x).series()


# In[157]:

(1/(1+x)).series()


# In[158]:

expr = sympy.cos(x) / (1 + sympy.sin(x * y))


# In[159]:

expr.series(x, n=4)


# In[160]:

expr.series(y, n=4)


# In[161]:

expr.series(y).removeO().series(x).removeO().expand()


# ## Limits

# In[162]:

sympy.limit(sympy.sin(x) / x, x, 0)


# In[163]:

f = sympy.Function('f')
x, h = sympy.symbols("x, h")


# In[164]:

diff_limit = (f(x + h) - f(x))/h


# In[165]:

sympy.limit(diff_limit.subs(f, sympy.cos), h, 0)


# In[166]:

sympy.limit(diff_limit.subs(f, sympy.sin), h, 0)


# In[167]:

expr = (x**2 - 3*x) / (2*x - 2)


# In[168]:

p = sympy.limit(expr/x, x, oo)


# In[169]:

q = sympy.limit(expr - p*x, x, oo)


# In[170]:

p, q


# ## Sums and products

# In[171]:

n = sympy.symbols("n", integer=True)


# In[172]:

x = sympy.Sum(1/(n**2), (n, 1, oo))


# In[173]:

x


# In[174]:

x.doit()


# In[175]:

x = sympy.Product(n, (n, 1, 7))


# In[176]:

x


# In[177]:

x.doit()


# In[178]:

x = sympy.Symbol("x")


# In[179]:

sympy.Sum((x)**n/(sympy.factorial(n)), (n, 1, oo)).doit().simplify()


# ## Equations

# In[180]:

x = sympy.symbols("x")


# In[181]:

sympy.solve(x**2 + 2*x - 3)


# In[182]:

a, b, c = sympy.symbols("a, b, c")


# In[183]:

sympy.solve(a * x**2 + b * x + c, x)


# In[184]:

sympy.solve(sympy.sin(x) - sympy.cos(x), x)


# In[185]:

sympy.solve(sympy.exp(x) + 2 * x, x)


# In[186]:

sympy.solve(x**5 - x**2 + 1, x)


# In[187]:

1 #s.solve(s.tan(x) - x, x)


# In[188]:

eq1 = x + 2 * y - 1
eq2 = x - y + 1


# In[189]:

sympy.solve([eq1, eq2], [x, y], dict=True)


# In[190]:

eq1 = x**2 - y
eq2 = y**2 - x


# In[191]:

sols = sympy.solve([eq1, eq2], [x, y], dict=True)


# In[192]:

sols


# In[193]:

[eq1.subs(sol).simplify() == 0 and eq2.subs(sol).simplify() == 0 for sol in sols]


# ## Linear algebra

# In[194]:

sympy.Matrix([1,2])


# In[195]:

sympy.Matrix([[1,2]])


# In[196]:

sympy.Matrix([[1, 2], [3, 4]])


# In[197]:

sympy.Matrix(3, 4, lambda m,n: 10 * m + n)


# In[198]:

a, b, c, d = sympy.symbols("a, b, c, d")


# In[199]:

M = sympy.Matrix([[a, b], [c, d]])


# In[200]:

M


# In[201]:

M * M


# In[202]:

x = sympy.Matrix(sympy.symbols("x_1, x_2"))


# In[203]:

M * x


# In[204]:

p, q = sympy.symbols("p, q")


# In[205]:

M = sympy.Matrix([[1, p], [q, 1]])


# In[206]:

M


# In[207]:

b = sympy.Matrix(sympy.symbols("b_1, b_2"))


# In[208]:

b


# In[209]:

x = M.solve(b)
x


# In[210]:

x = M.LUsolve(b)


# In[211]:

x


# In[212]:

x = M.inv() * b


# In[213]:

x


# ## Versions

# In[1]:

get_ipython().magic(u'reload_ext version_information')
get_ipython().magic(u'version_information sympy, numpy')

