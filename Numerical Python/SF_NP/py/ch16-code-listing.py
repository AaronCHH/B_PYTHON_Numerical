
# coding: utf-8

# # Chapter 16: Bayesian statistics

# Robert Johansson
# 
# Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).
# 
# The source code listings can be downloaded from http://www.apress.com/9781484205549

# In[1]:

import pymc3 as mc 


# In[2]:

import numpy as np


# In[3]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[4]:

import seaborn as sns


# In[5]:

from scipy import stats


# In[6]:

import statsmodels.api as sm


# In[7]:

import statsmodels.formula.api as smf


# In[8]:

import pandas as pd


# # Simple example: Normal distributed random variable

# In[9]:

# try this
# dir(mc.distributions)


# In[10]:

np.random.seed(100)


# In[11]:

mu = 4.0


# In[12]:

sigma = 2.0


# In[13]:

model = mc.Model()


# In[14]:

with model:
    mc.Normal('X', mu, 1/sigma**2)


# In[15]:

model.vars


# In[16]:

start = dict(X=2)


# In[17]:

with model:
    step = mc.Metropolis()
    trace = mc.sample(10000, step=step, start=start)


# In[18]:

x = np.linspace(-4, 12, 1000)


# In[19]:

y = stats.norm(mu, sigma).pdf(x)


# In[20]:

X = trace.get_values("X")


# In[21]:

fig, ax = plt.subplots(figsize=(8, 3))

ax.plot(x, y, 'r', lw=2)
sns.distplot(X, ax=ax)
ax.set_xlim(-4, 12)
ax.set_xlabel("x")
ax.set_ylabel("Probability distribution")
fig.tight_layout()
fig.savefig("ch16-normal-distribution-sampled.pdf")


# In[22]:

fig, axes = plt.subplots(1, 2, figsize=(8, 2.5), squeeze=False)
mc.traceplot(trace, ax=axes)
axes[0,0].plot(x, y, 'r', lw=0.5)
fig.tight_layout()
fig.savefig("ch16-normal-sampling-trace.png")
fig.savefig("ch16-normal-sampling-trace.pdf")


# ## Dependent random variables

# In[23]:

model = mc.Model()


# In[24]:

with model:
    mean = mc.Normal('mean', 3.0)
    sigma = mc.HalfNormal('sigma', sd=1.0)
    X = mc.Normal('X', mean, sd=sigma)


# In[25]:

model.vars


# In[26]:

with model:
    start = mc.find_MAP()


# In[27]:

start


# In[28]:

with model:
    step = mc.Metropolis()
    trace = mc.sample(100000, start=start, step=step)


# In[29]:

trace.get_values('sigma').mean()


# In[30]:

X = trace.get_values('X')


# In[31]:

X.mean()


# In[32]:

trace.get_values('X').std()


# In[33]:

fig, axes = plt.subplots(3, 2, figsize=(8, 6), squeeze=False)
mc.traceplot(trace, vars=['mean', 'sigma', 'X'], ax=axes)
fig.tight_layout()
fig.savefig("ch16-dependent-rv-sample-trace.png")
fig.savefig("ch16-dependent-rv-sample-trace.pdf")


# ## Posterior distributions

# In[34]:

mu = 2.5


# In[35]:

s = 1.5


# In[36]:

data = stats.norm(mu, s).rvs(100)


# In[37]:

with mc.Model() as model:
    
    mean = mc.Normal('mean', 4.0, 1.0) # true 2.5
    sigma = mc.HalfNormal('sigma', 3.0 * np.sqrt(np.pi/2)) # true 1.5

    X = mc.Normal('X', mean, 1/sigma**2, observed=data)


# In[38]:

model.vars


# In[39]:

with model:
    start = mc.find_MAP()
    step = mc.Metropolis()
    trace = mc.sample(100000, start=start, step=step)
    #step = mc.NUTS()
    #trace = mc.sample(10000, start=start, step=step)


# In[40]:

start


# In[41]:

model.vars


# In[42]:

fig, axes = plt.subplots(2, 2, figsize=(8, 4), squeeze=False)
mc.traceplot(trace, vars=['mean', 'sigma'], ax=axes)
fig.tight_layout()
fig.savefig("ch16-posterior-sample-trace.png")
fig.savefig("ch16-posterior-sample-trace.pdf")


# In[43]:

mu, trace.get_values('mean').mean()


# In[44]:

s, trace.get_values('sigma').mean()


# In[45]:

gs = mc.forestplot(trace, vars=['mean', 'sigma'])
plt.savefig("ch16-forestplot.pdf")


# In[46]:

help(mc.summary)


# In[47]:

mc.summary(trace, vars=['mean', 'sigma'])


# ## Linear regression

# In[48]:

dataset = sm.datasets.get_rdataset("Davis", "car")


# In[49]:

data = dataset.data[dataset.data.sex == 'M']


# In[50]:

data = data[data.weight < 110]


# In[51]:

data.head(3)


# In[52]:

model = smf.ols("height ~ weight", data=data)


# In[53]:

result = model.fit()


# In[54]:

print(result.summary())


# In[55]:

x = np.linspace(50, 110, 25)


# In[56]:

y = result.predict({"weight": x})


# In[57]:

fig, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.plot(data.weight, data.height, 'o')
ax.plot(x, y, color="blue")
ax.set_xlabel("weight")
ax.set_ylabel("height")
fig.tight_layout()
fig.savefig("ch16-linear-ols-fit.pdf")


# In[58]:

with mc.Model() as model:
    sigma = mc.Uniform('sigma', 0, 10)
    intercept = mc.Normal('intercept', 125, sd=30)
    beta = mc.Normal('beta', 0, sd=5)
    
    height_mu = intercept + beta * data.weight

    # likelihood function
    mc.Normal('height', mu=height_mu, sd=sigma, observed=data.height)

    # predict
    predict_height = mc.Normal('predict_height', mu=intercept + beta * x, sd=sigma, shape=len(x)) 


# In[59]:

model.vars


# In[60]:

with model:
    start = mc.find_MAP()
    step = mc.NUTS(state=start)
    trace = mc.sample(10000, step, start=start)


# In[61]:

model.vars


# In[62]:

fig, axes = plt.subplots(2, 2, figsize=(8, 4), squeeze=False)
mc.traceplot(trace, vars=['intercept', 'beta'], ax=axes)
fig.savefig("ch16-linear-model-sample-trace.pdf")
fig.savefig("ch16-linear-model-sample-trace.png")


# In[63]:

intercept = trace.get_values("intercept").mean()
intercept


# In[64]:

beta = trace.get_values("beta").mean()
beta


# In[65]:

result.params


# In[66]:

result.predict({"weight": 90})


# In[67]:

weight_index = np.where(x == 90)[0][0]


# In[68]:

trace.get_values("predict_height")[:, weight_index].mean()


# In[69]:

fig, ax = plt.subplots(figsize=(8, 3))

sns.distplot(trace.get_values("predict_height")[:, weight_index], ax=ax)
ax.set_xlim(150, 210)
ax.set_xlabel("height")
ax.set_ylabel("Probability distribution")
fig.tight_layout()
fig.savefig("ch16-linear-model-predict-cut.pdf")


# In[70]:

fig, ax = plt.subplots(1, 1, figsize=(8, 3))

for n in range(500, 2000, 1):
    intercept = trace.get_values("intercept")[n]
    beta = trace.get_values("beta")[n]
    ax.plot(x, intercept + beta * x, color='red', lw=0.25, alpha=0.05)

intercept = trace.get_values("intercept").mean()
beta = trace.get_values("beta").mean()
ax.plot(x, intercept + beta * x, color='k', label="Mean Bayesian prediction")

ax.plot(data.weight, data.height, 'o')
ax.plot(x, y, '--', color="blue", label="OLS prediction")
ax.set_xlabel("weight")
ax.set_ylabel("height")
ax.legend(loc=0)
fig.tight_layout()
fig.savefig("ch16-linear-model-fit.pdf")
fig.savefig("ch16-linear-model-fit.png")


# In[71]:

with mc.Model() as model:
    mc.glm.glm('height ~ weight', data)
    step = mc.NUTS()
    trace = mc.sample(2000, step)


# In[72]:

fig, axes = plt.subplots(3, 2, figsize=(8, 6), squeeze=False)
mc.traceplot(trace, vars=['Intercept', 'weight', 'sd'], ax=axes)
fig.tight_layout()
fig.savefig("ch16-glm-sample-trace.pdf")
fig.savefig("ch16-glm-sample-trace.png")


# ## Multilevel model

# In[73]:

dataset = sm.datasets.get_rdataset("Davis", "car")


# In[74]:

data = dataset.data.copy()
data = data[data.weight < 110]


# In[75]:

data["sex"] = data["sex"].apply(lambda x: 1 if x == "F" else 0)


# In[76]:

data.head()


# In[77]:

with mc.Model() as model:

    # heirarchical model: hyper priors
    #intercept_mu = mc.Normal("intercept_mu", 125)
    #intercept_sigma = 30.0 #mc.Uniform('intercept_sigma', lower=0, upper=50)
    #beta_mu = mc.Normal("beta_mu", 0.0)
    #beta_sigma = 5.0 #mc.Uniform('beta_sigma', lower=0, upper=10)
    
    # multilevel model: prior parameters
    intercept_mu, intercept_sigma = 125, 30
    beta_mu, beta_sigma = 0.0, 5.0
    
    # priors
    intercept = mc.Normal('intercept', intercept_mu, sd=intercept_sigma, shape=2)
    beta = mc.Normal('beta', beta_mu, sd=beta_sigma, shape=2)
    error = mc.Uniform('error', 0, 10)

    # model equation
    sex_idx = data.sex.values
    height_mu = intercept[sex_idx] + beta[sex_idx] * data.weight

    mc.Normal('height', mu=height_mu, sd=error, observed=data.height)


# In[78]:

model.vars


# In[79]:

with model:
    start = mc.find_MAP()
    step = mc.NUTS(state=start)
    hessian = mc.find_hessian(start)
    trace = mc.sample(5000, step, start=start)


# In[80]:

fig, axes = plt.subplots(3, 2, figsize=(8, 6), squeeze=False)
mc.traceplot(trace, vars=['intercept', 'beta', 'error'], ax=axes)
fig.tight_layout()
fig.savefig("ch16-multilevel-sample-trace.pdf")
fig.savefig("ch16-multilevel-sample-trace.png")


# In[81]:

intercept_m, intercept_f = trace.get_values('intercept').mean(axis=0)


# In[82]:

intercept = trace.get_values('intercept').mean()


# In[83]:

beta_m, beta_f = trace.get_values('beta').mean(axis=0)


# In[84]:

beta = trace.get_values('beta').mean()


# In[85]:

fig, ax = plt.subplots(1, 1, figsize=(8, 3))

mask_m = data.sex == 0
mask_f = data.sex == 1

ax.plot(data.weight[mask_m], data.height[mask_m], 'o', color="steelblue", label="male", alpha=0.5)
ax.plot(data.weight[mask_f], data.height[mask_f], 'o', color="green", label="female", alpha=0.5)

x = np.linspace(35, 110, 50)
ax.plot(x, intercept_m + x * beta_m, color="steelblue", label="model male group")
ax.plot(x, intercept_f + x * beta_f, color="green", label="model female group")
ax.plot(x, intercept + x * beta, color="black", label="model both groups")

ax.set_xlabel("weight")
ax.set_ylabel("height")
ax.legend(loc=0)
fig.tight_layout()
fig.savefig("ch16-multilevel-linear-model-fit.pdf")
fig.savefig("ch16-multilevel-linear-model-fit.png")


# In[86]:

trace.get_values('error').mean()


# # Version

# In[87]:

get_ipython().magic(u'reload_ext version_information')


# In[88]:

get_ipython().magic(u'version_information numpy, pandas, matplotlib, statsmodels, pymc3')

