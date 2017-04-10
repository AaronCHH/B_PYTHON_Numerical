
# coding: utf-8

# # Chapter 2: Vectors, matrices and multidimensional arrays

# Robert Johansson
# 
# Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).
# 
# The source code listings can be downloaded from http://www.apress.com/9781484205549

# In[1]:

import numpy as np


# ## The NumPy array object

# In[2]:

data = np.array([[1, 2], [3, 4], [5, 6]])


# In[3]:

type(data)


# In[4]:

data


# In[5]:

data.ndim


# In[6]:

data.shape


# In[7]:

data.size


# In[8]:

data.dtype


# In[9]:

data.nbytes


# ## Data types

# In[10]:

np.array([1, 2, 3], dtype=np.int)


# In[11]:

np.array([1, 2, 3], dtype=np.float)


# In[12]:

np.array([1, 2, 3], dtype=np.complex)


# In[13]:

data = np.array([1, 2, 3], dtype=np.float)


# In[14]:

data


# In[15]:

data.dtype


# In[16]:

data = np.array([1, 2, 3], dtype=np.int)


# In[17]:

data.dtype


# In[18]:

data


# In[19]:

data = np.array([1, 2, 3], dtype=np.float)


# In[20]:

data


# In[21]:

data.astype(np.int)


# In[22]:

d1 = np.array([1, 2, 3], dtype=float)


# In[23]:

d2 = np.array([1, 2, 3], dtype=complex)


# In[24]:

d1 + d2


# In[25]:

(d1 + d2).dtype


# In[26]:

np.sqrt(np.array([-1, 0, 1]))


# In[27]:

np.sqrt(np.array([-1, 0, 1], dtype=complex))


# ### Real and imaginary parts

# In[28]:

data = np.array([1, 2, 3], dtype=complex)


# In[29]:

data


# In[30]:

data.real


# In[31]:

data.imag


# ## Creating arrays

# ### Arrays created from lists and other array-like objects

# In[32]:

np.array([1, 2, 3, 4])


# In[33]:

data.ndim


# In[34]:

data.shape


# In[35]:

np.array([[1, 2], [3, 4]])


# In[36]:

data.ndim


# In[37]:

data.shape


# ### Arrays filled with constant values

# In[38]:

np.zeros((2, 3))


# In[39]:

np.ones(4)


# In[40]:

data = np.ones(4)


# In[41]:

data.dtype


# In[42]:

data = np.ones(4, dtype=np.int64)


# In[43]:

data.dtype


# In[44]:

x1 = 5.4 * np.ones(10)


# In[45]:

x2 = np.full(10, 5.4)


# In[46]:

x1 = np.empty(5)


# In[47]:

x1.fill(3.0)


# In[48]:

x1


# In[49]:

x2 = np.full(5, 3.0)


# In[50]:

x2


# ### Arrays filled with incremental sequences

# In[51]:

np.arange(0.0, 10, 1)


# In[52]:

np.linspace(0, 10, 11)


# ### Arrays filled with logarithmic sequences

# In[53]:

np.logspace(0, 2, 5)  # 5 data points between 10**0=1 to 10**2=100


# ### Mesh-grid arrays

# In[54]:

x = np.array([-1, 0, 1])


# In[55]:

y = np.array([-2, 0, 2])


# In[56]:

X, Y = np.meshgrid(x, y)


# In[57]:

X


# In[58]:

Y


# In[59]:

Z = (X + Y) ** 2


# In[60]:

Z


# ### Creating uninitialized arrays

# In[61]:

np.empty(3, dtype=np.float)


# ### Creating arrays with properties of other arrays

# In[62]:

def f(x):
    y = np.ones_like(x)
    # compute with x and y
    return y


# ### Creating matrix arrays

# In[63]:

np.identity(4)


# In[64]:

np.eye(3, k=1)


# In[65]:

np.eye(3, k=-1)


# In[66]:

np.diag(np.arange(0, 20, 5))


# ## Index and slicing

# ### One-dimensional arrays

# In[67]:

a = np.arange(0, 11)


# In[68]:

a


# In[69]:

a[0]  # the first element


# In[70]:

a[-1] # the last element


# In[71]:

a[4]  # the fifth element, at index 4


# In[72]:

a[1:-1]


# In[73]:

a[1:-1:2]


# In[74]:

a[:5]


# In[75]:

a[-5:]


# In[76]:

a[::-2]


# ## Multidimensional arrays

# In[77]:

f = lambda m, n: n + 10 * m


# In[78]:

A = np.fromfunction(f, (6, 6), dtype=int)


# In[79]:

A


# In[80]:

A[:, 1]  # the second column


# In[81]:

A[1, :]  # the second row


# In[82]:

A[:3, :3]  # upper half diagonal block matrix


# In[83]:

A[3:, :3]  # lower left off-diagonal block matrix


# In[84]:

A[::2, ::2]  # every second element starting from 0, 0


# In[85]:

A[1::2, 1::3]  # every second element starting from 1, 1


# ### Views

# In[86]:

B = A[1:5, 1:5]


# In[87]:

B


# In[88]:

B[:, :] = 0


# In[89]:

A


# In[90]:

C = B[1:3, 1:3].copy()


# In[91]:

C


# In[92]:

C[:, :] = 1  # this does not affect B since C is a copy of the view B[1:3, 1:3]


# In[93]:

C


# In[94]:

B


# ### Fancy indexing and Boolean-valued indexing

# In[95]:

A = np.linspace(0, 1, 11)


# In[96]:

A[np.array([0, 2, 4])]


# In[97]:

A[[0, 2, 4]]


# In[98]:

A > 0.5 


# In[99]:

A[A > 0.5]


# In[100]:

A = np.arange(10)


# In[101]:

A


# In[102]:

indices = [2, 4, 6]


# In[103]:

B = A[indices]


# In[104]:

B[0] = -1  # this does not affect A


# In[105]:

A


# In[106]:

A[indices] = -1


# In[107]:

A


# In[108]:

A = np.arange(10)


# In[109]:

B = A[A > 5]


# In[110]:

B[0] = -1  # this does not affect A


# In[111]:

A


# In[112]:

A[A > 5] = -1


# In[113]:

A


# ## Reshaping and resizing

# In[114]:

data = np.array([[1, 2], [3, 4]])


# In[115]:

np.reshape(data, (1, 4))


# In[116]:

data.reshape(4)


# In[117]:

data = np.array([[1, 2], [3, 4]])


# In[118]:

data


# In[119]:

data.flatten()


# In[120]:

data.flatten().shape


# In[121]:

data = np.arange(0, 5)


# In[122]:

column = data[:, np.newaxis]


# In[123]:

column


# In[124]:

row = data[np.newaxis, :]


# In[125]:

row


# In[126]:

data = np.arange(5)


# In[127]:

data


# In[128]:

np.vstack((data, data, data))


# In[129]:

data = np.arange(5)


# In[130]:

data


# In[131]:

np.hstack((data, data, data))


# In[132]:

data = data[:, np.newaxis]


# In[133]:

np.hstack((data, data, data))


# ## Vectorized expressions

# ### Arithmetic operations

# In[134]:

x = np.array([[1, 2], [3, 4]]) 


# In[135]:

y = np.array([[5, 6], [7, 8]])


# In[136]:

x + y


# In[137]:

y - x


# In[138]:

x * y


# In[139]:

y / x


# In[140]:

x * 2


# In[141]:

2 ** x


# In[142]:

y / 2


# In[143]:

(y / 2).dtype


# In[144]:

x = np.array([1, 2, 3, 4]).reshape(2,2)


# In[145]:

z = np.array([1, 2, 3, 4])


# In[146]:

x / z


# In[147]:

z = np.array([[2, 4]])


# In[148]:

z


# In[149]:

z.shape


# In[150]:

x / z


# In[151]:

zz = np.concatenate([z, z], axis=0)


# In[152]:

zz


# In[153]:

x / zz


# In[154]:

z = np.array([[2], [4]])


# In[155]:

z.shape


# In[156]:

x / z


# In[157]:

zz = np.concatenate([z, z], axis=1)


# In[158]:

zz


# In[159]:

x / zz


# In[160]:

x = np.array([[1, 3], [2, 4]])
x = x + y
x


# In[161]:

x = np.array([[1, 3], [2, 4]])
x += y
x


# ### Elementwise functions

# In[162]:

x = np.linspace(-1, 1, 11)


# In[163]:

x


# In[164]:

y = np.sin(np.pi * x)


# In[165]:

np.round(y, decimals=4)


# In[166]:

np.add(np.sin(x) ** 2, np.cos(x) ** 2)


# In[167]:

np.sin(x) ** 2 + np.cos(x) ** 2


# In[168]:

def heaviside(x):
    return 1 if x > 0 else 0


# In[169]:

heaviside(-1)


# In[170]:

heaviside(1.5)


# In[171]:

x = np.linspace(-5, 5, 11)


# In[172]:

heaviside(x)


# In[173]:

heaviside = np.vectorize(heaviside)


# In[174]:

heaviside(x)


# In[175]:

def heaviside(x):
    return 1.0 * (x > 0)


# ### Aggregate functions

# In[176]:

data = np.random.normal(size=(15,15))


# In[177]:

np.mean(data)


# In[178]:

data.mean()


# In[179]:

data = np.random.normal(size=(5, 10, 15))


# In[180]:

data.sum(axis=0).shape


# In[181]:

data.sum(axis=(0, 2)).shape


# In[182]:

data.sum()


# In[183]:

data = np.arange(1,10).reshape(3,3)


# In[184]:

data


# In[185]:

data.sum()


# In[186]:

data.sum(axis=0)


# In[187]:

data.sum(axis=1)


# ### Boolean arrays and conditional expressions

# In[188]:

a = np.array([1, 2, 3, 4])


# In[189]:

b = np.array([4, 3, 2, 1])


# In[190]:

a < b


# In[191]:

np.all(a < b)


# In[192]:

np.any(a < b)


# In[193]:

if np.all(a < b):
    print("All elements in a are smaller than their corresponding element in b")
elif np.any(a < b):
    print("Some elements in a are smaller than their corresponding elemment in b")
else:
    print("All elements in b are smaller than their corresponding element in a")


# In[194]:

x = np.array([-2, -1, 0, 1, 2])


# In[195]:

x > 0


# In[196]:

1 * (x > 0)


# In[197]:

x * (x > 0)


# In[198]:

def pulse(x, position, height, width):
    return height * (x >= position) * (x <= (position + width))


# In[199]:

x = np.linspace(-5, 5, 11)


# In[200]:

pulse(x, position=-2, height=1, width=5)


# In[201]:

pulse(x, position=1, height=1, width=5)


# In[202]:

def pulse(x, position, height, width):
    return height * np.logical_and(x >= position, x <= (position + width))


# In[203]:

x = np.linspace(-4, 4, 9)


# In[204]:

np.where(x < 0, x**2, x**3)


# In[205]:

np.select([x < -1, x < 2, x >= 2],
          [x**2  , x**3 , x**4])


# In[206]:

np.choose([0, 0, 0, 1, 1, 1, 2, 2, 2], 
          [x**2,    x**3,    x**4])


# In[207]:

x[abs(x) > 2]


# In[208]:

np.nonzero(abs(x) > 2)


# In[209]:

x[np.nonzero(abs(x) > 2)]


# # Set operations

# In[210]:

a = np.unique([1,2,3,3])


# In[211]:

b = np.unique([2,3,4,4,5,6,5])


# In[212]:

np.in1d(a, b)


# In[213]:

1 in a


# In[214]:

1 in b


# In[215]:

np.all(np.in1d(a, b))


# In[216]:

np.union1d(a, b)


# In[217]:

np.intersect1d(a, b)


# In[218]:

np.setdiff1d(a, b)


# In[219]:

np.setdiff1d(b, a)


# ### Operations on arrays

# In[220]:

data = np.arange(9).reshape(3, 3)


# In[221]:

data


# In[222]:

np.transpose(data)


# In[223]:

data = np.random.randn(1, 2, 3, 4, 5)


# In[224]:

data.shape


# In[225]:

data.T.shape


# ## Matrix and vector operations

# In[226]:

A = np.arange(1, 7).reshape(2, 3)


# In[227]:

A


# In[228]:

B = np.arange(1, 7).reshape(3, 2)


# In[229]:

B


# In[230]:

np.dot(A, B)


# In[231]:

np.dot(B, A)


# In[232]:

A = np.arange(9).reshape(3, 3)


# In[233]:

A


# In[234]:

x = np.arange(3)


# In[235]:

x


# In[236]:

np.dot(A, x)


# In[237]:

A.dot(x)


# In[238]:

A = np.random.rand(3,3)
B = np.random.rand(3,3)


# In[239]:

Ap = np.dot(B, np.dot(A, np.linalg.inv(B)))


# In[240]:

Ap = B.dot(A.dot(np.linalg.inv(B)))


# In[241]:

A = np.matrix(A)


# In[242]:

B = np.matrix(B)


# In[243]:

Ap = B * A * B.I


# In[244]:

A = np.asmatrix(A)


# In[245]:

B = np.asmatrix(B)


# In[246]:

Ap = B * A * B.I


# In[247]:

Ap = np.asarray(Ap)


# In[248]:

np.inner(x, x)


# In[249]:

np.dot(x, x)


# In[250]:

y = x[:, np.newaxis]


# In[251]:

y


# In[252]:

np.dot(y.T, y)


# In[253]:

x = np.array([1, 2, 3])


# In[254]:

np.outer(x, x) 


# In[255]:

np.kron(x, x) 


# In[256]:

np.kron(x[:, np.newaxis], x[np.newaxis, :])


# In[257]:

np.kron(np.ones((2,2)), np.identity(2))


# In[258]:

np.kron(np.identity(2), np.ones((2,2)))


# In[259]:

x = np.array([1, 2, 3, 4])


# In[260]:

y = np.array([5, 6, 7, 8])


# In[261]:

np.einsum("n,n", x, y)


# In[262]:

np.inner(x, y)


# In[263]:

A = np.arange(9).reshape(3, 3)


# In[264]:

B = A.T


# In[265]:

np.einsum("mk,kn", A, B)


# In[266]:

np.alltrue(np.einsum("mk,kn", A, B) == np.dot(A, B))


# # Versions

# In[267]:

get_ipython().magic(u'reload_ext version_information')
get_ipython().magic(u'version_information numpy')

