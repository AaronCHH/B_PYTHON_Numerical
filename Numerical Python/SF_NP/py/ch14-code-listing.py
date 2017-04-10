
# coding: utf-8

# # Chapter 14: Statistical modelling

# Robert Johansson
# 
# Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).
# 
# The source code listings can be downloaded from http://www.apress.com/9781484205549

# In[1]:

import statsmodels.api as sm


# In[2]:

import statsmodels.formula.api as smf


# In[3]:

import statsmodels.graphics.api as smg


# In[4]:

import patsy


# In[5]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[6]:

import numpy as np


# In[7]:

import pandas as pd


# In[8]:

from scipy import stats


# In[9]:

import seaborn as sns


# ## Statistical models and patsy formula

# In[10]:

np.random.seed(123456789)


# In[11]:

y = np.array([1, 2, 3, 4, 5])


# In[12]:

x1 = np.array([6, 7, 8, 9, 10])


# In[13]:

x2 = np.array([11, 12, 13, 14, 15])


# In[14]:

X = np.vstack([np.ones(5), x1, x2, x1*x2]).T


# In[15]:

X


# In[16]:

beta, res, rank, sval = np.linalg.lstsq(X, y)


# In[17]:

beta


# In[18]:

data = {"y": y, "x1": x1, "x2": x2}


# In[19]:

y, X = patsy.dmatrices("y ~ 1 + x1 + x2 + x1*x2", data)


# In[20]:

y


# In[21]:

X


# In[22]:

type(X)


# In[23]:

np.array(X)


# In[24]:

df_data = pd.DataFrame(data)


# In[25]:

y, X = patsy.dmatrices("y ~ 1 + x1 + x2 + x1:x2", df_data, return_type="dataframe")


# In[26]:

X


# In[27]:

model = sm.OLS(y, X)


# In[28]:

result = model.fit()


# In[29]:

result.params


# In[30]:

model = smf.ols("y ~ 1 + x1 + x2 + x1:x2", df_data)


# In[31]:

result = model.fit()


# In[32]:

result.params


# In[33]:

print(result.summary())


# In[34]:

beta


# In[35]:

from collections import defaultdict


# In[36]:

data = defaultdict(lambda: np.array([1,2,3]))


# In[37]:

patsy.dmatrices("y ~ a", data=data)[1].design_info.term_names


# In[38]:

patsy.dmatrices("y ~ 1 + a + b", data=data)[1].design_info.term_names


# In[39]:

patsy.dmatrices("y ~ -1 + a + b", data=data)[1].design_info.term_names


# In[40]:

patsy.dmatrices("y ~ a * b", data=data)[1].design_info.term_names


# In[41]:

patsy.dmatrices("y ~ a * b * c", data=data)[1].design_info.term_names


# In[42]:

patsy.dmatrices("y ~ a * b * c - a:b:c", data=data)[1].design_info.term_names


# In[43]:

data = {k: np.array([]) for k in ["y", "a", "b", "c"]}


# In[44]:

patsy.dmatrices("y ~ a + b", data=data)[1].design_info.term_names


# In[45]:

patsy.dmatrices("y ~ I(a + b)", data=data)[1].design_info.term_names


# In[46]:

patsy.dmatrices("y ~ a*a", data=data)[1].design_info.term_names


# In[47]:

patsy.dmatrices("y ~ I(a**2)", data=data)[1].design_info.term_names


# In[48]:

patsy.dmatrices("y ~ np.log(a) + b", data=data)[1].design_info.term_names


# In[49]:

z = lambda x1, x2: x1+x2


# In[50]:

patsy.dmatrices("y ~ z(a, b)", data=data)[1].design_info.term_names


# ### Categorical variables

# In[51]:

data = {"y": [1, 2, 3], "a": [1, 2, 3]}


# In[52]:

patsy.dmatrices("y ~ - 1 + a", data=data, return_type="dataframe")[1]


# In[53]:

patsy.dmatrices("y ~ - 1 + C(a)", data=data, return_type="dataframe")[1]


# In[54]:

data = {"y": [1, 2, 3], "a": ["type A", "type B", "type C"]}


# In[55]:

patsy.dmatrices("y ~ - 1 + a", data=data, return_type="dataframe")[1]


# In[56]:

patsy.dmatrices("y ~ - 1 + C(a, Poly)", data=data, return_type="dataframe")[1]


# # Linear regression

# In[57]:

np.random.seed(123456789)


# In[58]:

N = 100


# In[59]:

x1 = np.random.randn(N)


# In[60]:

x2 = np.random.randn(N)


# In[61]:

data = pd.DataFrame({"x1": x1, "x2": x2})


# In[62]:

def y_true(x1, x2):
    return 1  + 2 * x1 + 3 * x2 + 4 * x1 * x2


# In[63]:

data["y_true"] = y_true(x1, x2)


# In[64]:

e = np.random.randn(N)


# In[65]:

data["y"] = data["y_true"] + e


# In[66]:

data.head()


# In[67]:

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].scatter(data["x1"], data["y"])
axes[1].scatter(data["x2"], data["y"])

fig.tight_layout()


# In[68]:

data.shape


# In[69]:

model = smf.ols("y ~ x1 + x2", data)


# In[70]:

result = model.fit()


# In[71]:

print(result.summary())


# In[72]:

result.rsquared


# In[73]:

result.resid.head()


# In[74]:

z, p = stats.normaltest(result.resid.values)


# In[75]:

p


# In[76]:

result.params


# In[77]:

fig, ax = plt.subplots(figsize=(8, 4))
smg.qqplot(result.resid, ax=ax)

fig.tight_layout()
fig.savefig("ch14-qqplot-model-1.pdf")


# In[78]:

model = smf.ols("y ~ x1 + x2 + x1*x2", data)


# In[79]:

result = model.fit()


# In[80]:

print(result.summary())


# In[81]:

result.params


# In[82]:

result.rsquared


# In[83]:

z, p = stats.normaltest(result.resid.values)


# In[84]:

p


# In[85]:

fig, ax = plt.subplots(figsize=(8, 4))
smg.qqplot(result.resid, ax=ax)

fig.tight_layout()
fig.savefig("ch14-qqplot-model-2.pdf")


# In[86]:

x = np.linspace(-1, 1, 50)


# In[87]:

X1, X2 = np.meshgrid(x, x)


# In[88]:

new_data = pd.DataFrame({"x1": X1.ravel(), "x2": X2.ravel()})


# In[89]:

y_pred = result.predict(new_data)


# In[90]:

y_pred.shape


# In[91]:

y_pred = y_pred.reshape(50, 50)


# In[92]:

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

def plot_y_contour(ax, Y, title):
    c = ax.contourf(X1, X2, Y, 15, cmap=plt.cm.RdBu)
    ax.set_xlabel(r"$x_1$", fontsize=20)
    ax.set_ylabel(r"$x_2$", fontsize=20)
    ax.set_title(title)
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(r"$y$", fontsize=20)

plot_y_contour(axes[0], y_true(X1, X2), "true relation")
plot_y_contour(axes[1], y_pred, "fitted model")

fig.tight_layout()
fig.savefig("ch14-comparison-model-true.pdf")


# ### Datasets from R

# In[93]:

dataset = sm.datasets.get_rdataset("Icecream", "Ecdat")


# In[94]:

dataset.title


# In[95]:

print(dataset.__doc__)


# In[96]:

dataset.data.info()


# In[97]:

model = smf.ols("cons ~ -1 + price + temp", data=dataset.data)


# In[98]:

result = model.fit()


# In[99]:

print(result.summary())


# In[100]:

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

smg.plot_fit(result, 0, ax=ax1)
smg.plot_fit(result, 1, ax=ax2)

fig.tight_layout()
fig.savefig("ch14-regressionplots.pdf")


# In[101]:

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

sns.regplot("price", "cons", dataset.data, ax=ax1);
sns.regplot("temp", "cons", dataset.data, ax=ax2);

fig.tight_layout()
fig.savefig("ch14-regressionplots-seaborn.pdf")


# ## Discrete regression, logistic regression

# In[102]:

df = sm.datasets.get_rdataset("iris").data


# In[103]:

df.info()


# In[104]:

df.Species.unique()


# In[105]:

df_subset = df[(df.Species == "versicolor") | (df.Species == "virginica" )].copy()


# In[106]:

df_subset.Species = df_subset.Species.map({"versicolor": 1, "virginica": 0})


# In[107]:

df_subset.rename(columns={"Sepal.Length": "Sepal_Length", "Sepal.Width": "Sepal_Width",
                          "Petal.Length": "Petal_Length", "Petal.Width": "Petal_Width"}, inplace=True)


# In[108]:

df_subset.head(3)


# In[109]:

model = smf.logit("Species ~ Sepal_Length + Sepal_Width + Petal_Length + Petal_Width", data=df_subset)


# In[110]:

result = model.fit()


# In[111]:

print(result.summary())


# In[112]:

print(result.get_margeff().summary())


# **Note:** Sepal_Length and Sepal_Width do not seem to contribute much to predictiveness of the model. 

# In[113]:

model = smf.logit("Species ~ Petal_Length + Petal_Width", data=df_subset)


# In[114]:

result = model.fit()


# In[115]:

print(result.summary())


# In[116]:

print(result.get_margeff().summary())


# In[117]:

params = result.params
beta0 = -params['Intercept']/params['Petal_Width']
beta1 = -params['Petal_Length']/params['Petal_Width']


# In[118]:

df_new = pd.DataFrame({"Petal_Length": np.random.randn(20)*0.5 + 5,
                       "Petal_Width": np.random.randn(20)*0.5 + 1.7})


# In[119]:

df_new["P-Species"] = result.predict(df_new)


# In[120]:

df_new["P-Species"].head(3)


# In[121]:

df_new["Species"] = (df_new["P-Species"] > 0.5).astype(int)


# In[122]:

df_new.head()


# In[123]:

fig, ax = plt.subplots(1, 1, figsize=(8, 4))

ax.plot(df_subset[df_subset.Species == 0].Petal_Length.values,
        df_subset[df_subset.Species == 0].Petal_Width.values, 's', label='virginica')
ax.plot(df_new[df_new.Species == 0].Petal_Length.values,
        df_new[df_new.Species == 0].Petal_Width.values,
        'o', markersize=10, color="steelblue", label='virginica (pred.)')

ax.plot(df_subset[df_subset.Species == 1].Petal_Length.values,
        df_subset[df_subset.Species == 1].Petal_Width.values, 's', label='versicolor')
ax.plot(df_new[df_new.Species == 1].Petal_Length.values,
        df_new[df_new.Species == 1].Petal_Width.values,
        'o', markersize=10, color="green", label='versicolor (pred.)')

_x = np.array([4.0, 6.1])
ax.plot(_x, beta0 + beta1 * _x, 'k')

ax.set_xlabel('Petal length')
ax.set_ylabel('Petal width')
ax.legend(loc=2)
fig.tight_layout()
fig.savefig("ch14-logit.pdf")


# ### Poisson distribution

# In[124]:

dataset = sm.datasets.get_rdataset("discoveries")


# In[125]:

df = dataset.data.set_index("time")


# In[126]:

df.head(10).T


# In[127]:

fig, ax = plt.subplots(1, 1, figsize=(16, 4))
df.plot(kind='bar', ax=ax)
fig.tight_layout()
fig.savefig("ch14-discoveries.pdf")


# In[128]:

model = smf.poisson("discoveries ~ 1", data=df)


# In[129]:

result = model.fit()


# In[130]:

print(result.summary())


# In[131]:

lmbda = np.exp(result.params) 


# In[132]:

X = stats.poisson(lmbda)


# In[133]:

result.conf_int()


# In[134]:

X_ci_l = stats.poisson(np.exp(result.conf_int().values)[0, 0])


# In[135]:

X_ci_u = stats.poisson(np.exp(result.conf_int().values)[0, 1])


# In[136]:

v, k = np.histogram(df.values, bins=12, range=(0, 12), normed=True)


# In[137]:

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.bar(k[:-1], v, color="steelblue",  align='center', label='Dicoveries per year') 
ax.bar(k-0.125, X_ci_l.pmf(k), color="red", alpha=0.5, align='center', width=0.25, label='Poisson fit (CI, lower)')
ax.bar(k, X.pmf(k), color="green",  align='center', width=0.5, label='Poisson fit')
ax.bar(k+0.125, X_ci_u.pmf(k), color="red",  alpha=0.5, align='center', width=0.25, label='Poisson fit (CI, upper)')

ax.legend()
fig.tight_layout()
fig.savefig("ch14-discoveries-per-year.pdf")


# ## Time series

# In[138]:

df = pd.read_csv("temperature_outdoor_2014.tsv", header=None, delimiter="\t", names=["time", "temp"])
df.time = pd.to_datetime(df.time, unit="s")
df = df.set_index("time").resample("H")


# In[139]:

df_march = df[df.index.month == 3]


# In[140]:

df_april = df[df.index.month == 4]


# In[141]:

df_march.plot(figsize=(12, 4));


# In[142]:

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
smg.tsa.plot_acf(df_march.temp, lags=72, ax=axes[0])
smg.tsa.plot_acf(df_march.temp.diff().dropna(), lags=72, ax=axes[1])
smg.tsa.plot_acf(df_march.temp.diff().diff().dropna(), lags=72, ax=axes[2])
smg.tsa.plot_acf(df_march.temp.diff().diff().diff().dropna(), lags=72, ax=axes[3])
fig.tight_layout()
fig.savefig("ch14-timeseries-autocorrelation.pdf")


# In[143]:

model = sm.tsa.AR(df_march.temp)


# In[144]:

result = model.fit(72)


# In[145]:

sm.stats.durbin_watson(result.resid)


# In[146]:

fig, ax = plt.subplots(1, 1, figsize=(8, 3))
smg.tsa.plot_acf(result.resid, lags=72, ax=ax)
fig.tight_layout()
fig.savefig("ch14-timeseries-resid-acf.pdf")


# In[147]:

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(df_march.index.values[-72:], df_march.temp.values[-72:], label="train data")
ax.plot(df_april.index.values[:72], df_april.temp.values[:72], label="actual outcome")
ax.plot(pd.date_range("2014-04-01", "2014-04-4", freq="H").values,
        result.predict("2014-04-01", "2014-04-4"), label="predicted outcome")

ax.legend()
fig.tight_layout()
fig.savefig("ch14-timeseries-prediction.pdf")


# In[148]:

# Using ARMA model on daily average temperatures


# In[149]:

df_march = df_march.resample("D")


# In[150]:

df_april = df_april.resample("D")


# In[151]:

model = sm.tsa.ARMA(df_march, (4, 1))


# In[152]:

result = model.fit()


# In[153]:

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.plot(df_march.index.values[-3:], df_march.temp.values[-3:], 's-', label="train data")
ax.plot(df_april.index.values[:3], df_april.temp.values[:3], 's-', label="actual outcome")
ax.plot(pd.date_range("2014-04-01", "2014-04-3").values,
        result.predict("2014-04-01", "2014-04-3"), 's-', label="predicted outcome")
ax.legend()
fig.tight_layout()


# # Versions

# In[154]:

get_ipython().magic(u'reload_ext version_information')


# In[155]:

get_ipython().magic(u'version_information numpy, matplotlib, pandas, scipy, statsmodels, patsy')

