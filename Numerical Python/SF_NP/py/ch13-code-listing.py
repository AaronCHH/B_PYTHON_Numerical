
# coding: utf-8

# # Chapter 13: Statistics

# Robert Johansson
# 
# Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).
# 
# The source code listings can be downloaded from http://www.apress.com/9781484205549

# ## Imports

# In[1]:

from scipy import stats


# In[2]:

from scipy import optimize


# In[3]:

import numpy as np
import random


# In[4]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[5]:

import seaborn as sns


# In[6]:

sns.set(style="whitegrid")


# ## Descriptive statistics

# In[7]:

x = np.array([3.5, 1.1, 3.2, 2.8, 6.7, 4.4, 0.9, 2.2])


# In[8]:

np.mean(x)


# In[9]:

np.median(x)


# In[10]:

x.min(), x.max()


# In[11]:

x.var()


# In[12]:

x.std()


# In[13]:

x.var(ddof=1)


# In[14]:

x.std(ddof=1)


# ## Random numbers

# In[15]:

random.seed(123456789)


# In[16]:

random.random()


# In[17]:

random.randint(0, 10)  # 0 and 10 inclusive


# In[18]:

np.random.seed(123456789)


# In[19]:

np.random.rand()


# In[20]:

np.random.randn()


# In[21]:

np.random.rand(5)


# In[22]:

np.random.randn(2, 4)


# In[23]:

np.random.randint(10, size=10)


# In[24]:

np.random.randint(low=10, high=20, size=(2, 10))


# In[25]:

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

axes[0].hist(np.random.rand(10000))
axes[0].set_title("rand")
axes[1].hist(np.random.randn(10000))
axes[1].set_title("randn")
axes[2].hist(np.random.randint(low=1, high=10, size=10000), bins=9, align='left')
axes[2].set_title("randint(low=1, high=10)")

fig.tight_layout()
fig.savefig("ch13-random-hist.pdf")


# In[26]:

#random.sample(range(10), 5)


# In[27]:

np.random.choice(10, 5, replace=False)


# In[28]:

np.random.seed(123456789)


# In[29]:

np.random.rand()


# In[30]:

np.random.seed(123456789); np.random.rand()


# In[31]:

np.random.seed(123456789); np.random.rand()


# In[32]:

prng = np.random.RandomState(123456789)


# In[33]:

prng.rand(2, 4)


# In[34]:

prng.chisquare(1, size=(2, 2))


# In[35]:

prng.standard_t(1, size=(2, 3))


# In[36]:

prng.f(5, 2, size=(2, 4))


# In[37]:

prng.binomial(10, 0.5, size=10)


# In[38]:

prng.poisson(5, size=10)


# # Probability distributions and random variables

# In[39]:

np.random.seed(123456789)


# In[40]:

X = stats.norm(1, 0.5)


# In[41]:

X.mean()


# In[42]:

X.median()


# In[43]:

X.std()


# In[44]:

X.var()


# In[45]:

[X.moment(n) for n in range(5)]


# In[46]:

X.stats()


# In[47]:

X.pdf([0, 1, 2])


# In[48]:

X.cdf([0, 1, 2])


# In[49]:

X.rvs(10)


# In[50]:

stats.norm(1, 0.5).stats()


# In[51]:

stats.norm.stats(loc=2, scale=0.5)


# In[52]:

X.interval(0.95)


# In[53]:

X.interval(0.99)


# In[54]:

def plot_rv_distribution(X, axes=None):
    """Plot the PDF, CDF, SF and PPF of a given random variable"""
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    
    x_min_999, x_max_999 = X.interval(0.999)
    x999 = np.linspace(x_min_999, x_max_999, 1000)

    x_min_95, x_max_95 = X.interval(0.95)
    x95 = np.linspace(x_min_95, x_max_95, 1000)

    if hasattr(X.dist, 'pdf'):
        axes[0].plot(x999, X.pdf(x999), label="PDF")
        axes[0].fill_between(x95, X.pdf(x95), alpha=0.25)
    else:
        x999_int = np.unique(x999.astype(int))
        axes[0].bar(x999_int, X.pmf(x999_int), label="PMF")
    axes[1].plot(x999, X.cdf(x999), label="CDF")
    axes[1].plot(x999, X.sf(x999), label="SF")
    axes[2].plot(x999, X.ppf(x999), label="PPF")
    
    for ax in axes:
        ax.legend()
    
    return axes


# In[55]:

fig, axes = plt.subplots(3, 3, figsize=(12, 9))

X = stats.norm()
plot_rv_distribution(X, axes=axes[0, :])
axes[0, 0].set_ylabel("Normal dist.")
X = stats.f(2, 50)
plot_rv_distribution(X, axes=axes[1, :])
axes[1, 0].set_ylabel("F dist.")
X = stats.poisson(5)
plot_rv_distribution(X, axes=axes[2, :])
axes[2, 0].set_ylabel("Poisson dist.")

fig.tight_layout()
fig.savefig("ch13-distributions.pdf")


# In[56]:

def plot_dist_samples(X, X_samples, title=None, ax=None):
    """ Plot the PDF and histogram of samples of a continuous random variable"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    x_lim = X.interval(.99)
    x = np.linspace(*x_lim, num=100)

    ax.plot(x, X.pdf(x), label="PDF", lw=3)    
    ax.hist(X_samples, label="samples", normed=1, bins=75)
    ax.set_xlim(*x_lim)
    ax.legend()
    
    if title:
        ax.set_title(title)
    return ax


# In[57]:

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
X = stats.t(7.0)
plot_dist_samples(X, X.rvs(2000), "Student's t dist.", ax=axes[0])
X = stats.chi2(5.0)
plot_dist_samples(X, X.rvs(2000), r"$\chi^2$ dist.", ax=axes[1])
X = stats.expon(0.5)
plot_dist_samples(X, X.rvs(2000), "exponential dist.", ax=axes[2])
fig.tight_layout()
fig.savefig("ch13-dist-sample.pdf")


# In[58]:

X = stats.chi2(df=5)


# In[59]:

X_samples = X.rvs(500)


# In[60]:

df, loc, scale = stats.chi2.fit(X_samples)


# In[61]:

df, loc, scale


# In[62]:

Y = stats.chi2(df=df, loc=loc, scale=scale)


# In[63]:

fig, ax = plt.subplots(1, 1, figsize=(8, 3))

x_lim = X.interval(.99)
x = np.linspace(*x_lim, num=100)

ax.plot(x, X.pdf(x), label="original")
ax.plot(x, Y.pdf(x), label="recreated")
ax.legend()

fig.tight_layout()
fig.savefig("ch13-max-likelihood-fit.pdf")


# In[64]:

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

x_lim = X.interval(.99)
x = np.linspace(*x_lim, num=100)

axes[0].plot(x, X.pdf(x), label="original")
axes[0].plot(x, Y.pdf(x), label="recreated")
axes[0].legend()

axes[1].plot(x, X.pdf(x) - Y.pdf(x), label="error")
axes[1].legend()

fig.tight_layout()
fig.savefig("ch13-max-likelihood-fit.pdf")


# # Hypothesis testing

# In[65]:

np.random.seed(123456789)


# In[66]:

mu, sigma = 1.0, 0.5


# In[67]:

X = stats.norm(mu-0.2, sigma)


# In[68]:

n = 100


# In[69]:

X_samples = X.rvs(n)


# In[70]:

z = (X_samples.mean() - mu)/(sigma/np.sqrt(n))


# In[71]:

z


# In[72]:

t = (X_samples.mean() - mu)/(X_samples.std(ddof=1)/np.sqrt(n))


# In[73]:

t


# In[74]:

stats.norm().ppf(0.025)


# In[75]:

2 * stats.norm().cdf(-abs(z))


# In[76]:

2 * stats.t(df=(n-1)).cdf(-abs(t))


# In[77]:

t, p = stats.ttest_1samp(X_samples, mu)


# In[78]:

t


# In[79]:

p


# In[80]:

fig, ax = plt.subplots(figsize=(8, 3))

sns.distplot(X_samples, ax=ax)
x = np.linspace(*X.interval(0.999), num=100)
ax.plot(x, stats.norm(loc=mu, scale=sigma).pdf(x))

fig.tight_layout()
fig.savefig("ch13-hypothesis-test-dist-sample-mean.pdf")


# In[81]:

n = 50


# In[82]:

mu1, mu2 = np.random.rand(2)


# In[83]:

X1 = stats.norm(mu1, sigma)


# In[84]:

X1_sample = X1.rvs(n)


# In[85]:

X2 = stats.norm(mu2, sigma)


# In[86]:

X2_sample = X2.rvs(n)


# In[87]:

t, p = stats.ttest_ind(X1_sample, X2_sample)


# In[88]:

t


# In[89]:

p


# In[90]:

mu1, mu2


# In[91]:

sns.distplot(X1_sample)
sns.distplot(X2_sample)


# # Nonparameteric methods

# In[92]:

np.random.seed(0)


# In[93]:

X = stats.chi2(df=5)


# In[94]:

X_samples = X.rvs(100)


# In[95]:

kde = stats.kde.gaussian_kde(X_samples)


# In[96]:

kde_low_bw = stats.kde.gaussian_kde(X_samples, bw_method=0.25)


# In[97]:

x = np.linspace(0, 20, 100)


# In[98]:

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

axes[0].hist(X_samples, normed=True, alpha=0.5, bins=25)
axes[1].plot(x, kde(x), label="KDE")
axes[1].plot(x, kde_low_bw(x), label="KDE (low bw)")
axes[1].plot(x, X.pdf(x), label="True PDF")
axes[1].legend()
sns.distplot(X_samples, bins=25, ax=axes[2])

fig.tight_layout()
fig.savefig("ch13-hist-kde.pdf")


# In[99]:

kde.resample(10)


# In[100]:

def _kde_cdf(x):
    return kde.integrate_box_1d(-np.inf, x)


# In[101]:

kde_cdf = np.vectorize(_kde_cdf)


# In[102]:

fig, ax = plt.subplots(1, 1, figsize=(8, 3))

sns.distplot(X_samples, bins=25, ax=ax)
x = np.linspace(0, 20, 100)
ax.plot(x, kde_cdf(x))

fig.tight_layout()


# In[103]:

def _kde_ppf(q):
    return optimize.fsolve(lambda x, q: kde_cdf(x) - q, kde.dataset.mean(), args=(q,))[0]


# In[104]:

kde_ppf = np.vectorize(_kde_ppf)


# In[105]:

kde_ppf([0.05, 0.95])


# In[106]:

X.ppf([0.05, 0.95])


# # Versions

# In[107]:

get_ipython().magic(u'reload_ext version_information')


# In[108]:

get_ipython().magic(u'version_information numpy, scipy, matplotlib, seaborn')

