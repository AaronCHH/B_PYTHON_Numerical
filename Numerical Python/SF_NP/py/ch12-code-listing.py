
# coding: utf-8

# # Chapter 12: Data processing and analysis with `pandas`

# Robert Johansson
# 
# Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).
# 
# The source code listings can be downloaded from http://www.apress.com/9781484205549

# In[1]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[2]:

import numpy as np


# In[3]:

import pandas as pd


# In[4]:

pd.set_option('display.mpl_style', 'default')


# ## Series object

# In[5]:

s = pd.Series([909976, 8615246, 2872086, 2273305])


# In[6]:

s


# In[7]:

type(s)


# In[8]:

s.dtype


# In[9]:

s.index


# In[10]:

s.values


# In[11]:

s.index = ["Stockholm", "London", "Rome", "Paris"]


# In[12]:

s.name = "Population"


# In[13]:

s


# In[14]:

s = pd.Series([909976, 8615246, 2872086, 2273305], 
              index=["Stockholm", "London", "Rome", "Paris"], name="Population")


# In[15]:

s["London"]


# In[16]:

s.Stockholm


# In[17]:

s[["Paris", "Rome"]]


# In[18]:

s.median(), s.mean(), s.std()


# In[19]:

s.min(), s.max()


# In[20]:

s.quantile(q=0.25), s.quantile(q=0.5), s.quantile(q=0.75)


# In[21]:

s.describe()


# In[22]:

fig, axes = plt.subplots(1, 4, figsize=(12, 3))

s.plot(ax=axes[0], kind='line', title="line")
s.plot(ax=axes[1], kind='bar', title="bar")
s.plot(ax=axes[2], kind='box', title="box")
s.plot(ax=axes[3], kind='pie', title="pie")

fig.tight_layout()
fig.savefig("ch12-series-plot.pdf")
fig.savefig("ch12-series-plot.png")


# ## DataFrame object

# In[23]:

df = pd.DataFrame([[909976, 8615246, 2872086, 2273305],
                   ["Sweden", "United kingdom", "Italy", "France"]])


# In[24]:

df


# In[25]:

df = pd.DataFrame([[909976, "Sweden"],
                   [8615246, "United kingdom"], 
                   [2872086, "Italy"],
                   [2273305, "France"]])


# In[26]:

df


# In[27]:

df.index = ["Stockholm", "London", "Rome", "Paris"]


# In[28]:

df.columns = ["Population", "State"]


# In[29]:

df


# In[30]:

df = pd.DataFrame([[909976, "Sweden"],
                   [8615246, "United kingdom"], 
                   [2872086, "Italy"],
                   [2273305, "France"]],
                  index=["Stockholm", "London", "Rome", "Paris"],
                  columns=["Population", "State"])


# In[31]:

df


# In[32]:

df = pd.DataFrame({"Population": [909976, 8615246, 2872086, 2273305],
                   "State": ["Sweden", "United kingdom", "Italy", "France"]},
                  index=["Stockholm", "London", "Rome", "Paris"])


# In[33]:

df


# In[34]:

df.index


# In[35]:

df.columns


# In[36]:

df.values


# In[37]:

df.Population


# In[38]:

df["Population"]


# In[39]:

type(df.Population)


# In[40]:

df.Population.Stockholm


# In[41]:

type(df.ix)


# In[42]:

df.ix["Stockholm"]


# In[43]:

type(df.ix["Stockholm"])


# In[44]:

df.ix[["Paris", "Rome"]]


# In[45]:

df.ix[["Paris", "Rome"], "Population"]


# In[46]:

df.ix["Paris", "Population"]


# In[47]:

df.mean()


# In[48]:

df.info()


# In[49]:

df.dtypes


# In[50]:

df.head()


# In[51]:

get_ipython().system(u'head -n5 /home/rob/datasets/european_cities.csv')


# ## Larger dataset

# In[52]:

df_pop = pd.read_csv("european_cities.csv")


# In[53]:

df_pop.head()


# In[54]:

df_pop = pd.read_csv("european_cities.csv", delimiter=",", encoding="utf-8", header=0)


# In[55]:

df_pop.info()


# In[56]:

df_pop.head()


# In[57]:

df_pop["NumericPopulation"] = df_pop.Population.apply(lambda x: int(x.replace(",", "")))


# In[58]:

df_pop["State"].values[:3]


# In[59]:

df_pop["State"] = df_pop["State"].apply(lambda x: x.strip())


# In[60]:

df_pop.head()


# In[61]:

df_pop.dtypes


# In[62]:

df_pop2 = df_pop.set_index("City")


# In[63]:

df_pop2 = df_pop2.sort_index()


# In[64]:

df_pop2.head()


# In[65]:

df_pop2.head()


# In[66]:

df_pop3 = df_pop.set_index(["State", "City"]).sortlevel(0)


# In[67]:

df_pop3.head(7)


# In[68]:

df_pop3.ix["Sweden"]


# In[69]:

df_pop3.ix[("Sweden", "Gothenburg")]


# In[70]:

df_pop.set_index("City").sort(["State", "NumericPopulation"], ascending=[False, True]).head()


# In[71]:

city_counts = df_pop.State.value_counts()


# In[72]:

city_counts.name = "# cities in top 105"


# In[73]:

df_pop3 = df_pop[["State", "City", "NumericPopulation"]].set_index(["State", "City"])


# In[74]:

df_pop4 = df_pop3.sum(level="State").sort("NumericPopulation", ascending=False)


# In[75]:

df_pop4.head()


# In[76]:

df_pop5 = (df_pop.drop("Rank", axis=1)
                 .groupby("State").sum()
                 .sort("NumericPopulation", ascending=False))


# In[77]:

df_pop5.head()


# In[78]:

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

city_counts.plot(kind='barh', ax=ax1)
ax1.set_xlabel("# cities in top 105")
df_pop5.NumericPopulation.plot(kind='barh', ax=ax2)
ax2.set_xlabel("Total pop. in top 105 cities")

fig.tight_layout()
fig.savefig("ch12-state-city-counts-sum.pdf")


# ## Time series

# ### Basics

# In[79]:

import datetime


# In[80]:

pd.date_range("2015-1-1", periods=31)


# In[81]:

pd.date_range(datetime.datetime(2015, 1, 1), periods=31)


# In[82]:

pd.date_range("2015-1-1 00:00", "2015-1-1 12:00", freq="H")


# In[83]:

ts1 = pd.Series(np.arange(31), index=pd.date_range("2015-1-1", periods=31))


# In[84]:

ts1.head()


# In[85]:

ts1["2015-1-3"]


# In[86]:

ts1.index[2]


# In[87]:

ts1.index[2].year, ts1.index[2].month, ts1.index[2].day


# In[88]:

ts1.index[2].nanosecond


# In[89]:

ts1.index[2].to_pydatetime()


# In[90]:

ts2 = pd.Series(np.random.rand(2), 
                index=[datetime.datetime(2015, 1, 1), datetime.datetime(2015, 2, 1)])


# In[91]:

ts2


# In[92]:

periods = pd.PeriodIndex([pd.Period('2015-01'), pd.Period('2015-02'), pd.Period('2015-03')])


# In[93]:

ts3 = pd.Series(np.random.rand(3), periods)


# In[94]:

ts3


# In[95]:

ts3.index


# In[96]:

ts2.to_period('M')


# In[97]:

pd.date_range("2015-1-1", periods=12, freq="M").to_period()


# ### Temperature time series example

# In[98]:

get_ipython().system(u'head -n 5 temperature_outdoor_2014.tsv')


# In[99]:

df1 = pd.read_csv('temperature_outdoor_2014.tsv', delimiter="\t", names=["time", "outdoor"])


# In[100]:

df2 = pd.read_csv('temperature_indoor_2014.tsv', delimiter="\t", names=["time", "indoor"])


# In[101]:

df1.head()


# In[102]:

df2.head()


# In[103]:

df1.time = (pd.to_datetime(df1.time.values, unit="s")
              .tz_localize('UTC').tz_convert('Europe/Stockholm'))


# In[104]:

df1 = df1.set_index("time")


# In[105]:

df2.time = (pd.to_datetime(df2.time.values, unit="s")
              .tz_localize('UTC').tz_convert('Europe/Stockholm'))


# In[106]:

df2 = df2.set_index("time")


# In[107]:

df1.head()


# In[108]:

df1.index[0]


# In[109]:

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
df1.plot(ax=ax)
df2.plot(ax=ax)

fig.tight_layout()
fig.savefig("ch12-timeseries-temperature-2014.pdf")


# In[110]:

# select january data


# In[111]:

df1.info()


# In[112]:

df1_jan = df1[(df1.index > "2014-1-1") & (df1.index < "2014-2-1")]


# In[113]:

df1.index < "2014-2-1"


# In[114]:

df1_jan.info()


# In[115]:

df2_jan = df2["2014-1-1":"2014-1-31"]


# In[116]:

fig, ax = plt.subplots(1, 1, figsize=(12, 4))

df1_jan.plot(ax=ax)
df2_jan.plot(ax=ax)

fig.tight_layout()
fig.savefig("ch12-timeseries-selected-month.pdf")


# In[117]:

# group by month


# In[118]:

df1_month = df1.reset_index()


# In[119]:

df1_month["month"] = df1_month.time.apply(lambda x: x.month)


# In[120]:

df1_month.head()


# In[121]:

df1_month = df1_month.groupby("month").aggregate(np.mean)


# In[122]:

df2_month = df2.reset_index()


# In[123]:

df2_month["month"] = df2_month.time.apply(lambda x: x.month)


# In[124]:

df2_month = df2_month.groupby("month").aggregate(np.mean)


# In[125]:

df_month = df1_month.join(df2_month)


# In[126]:

df_month.head(3)


# In[127]:

df_month = pd.concat([df.to_period("M").groupby(level=0).mean() for df in [df1, df2]], axis=1)


# In[128]:

df_month.head(3)


# In[129]:

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df_month.plot(kind='bar', ax=axes[0])
df_month.plot(kind='box', ax=axes[1])

fig.tight_layout()
fig.savefig("ch12-grouped-by-month.pdf")


# In[130]:

df_month


# In[131]:

# resampling


# In[132]:

df1_hour = df1.resample("H")


# In[133]:

df1_hour.columns = ["outdoor (hourly avg.)"]


# In[134]:

df1_day = df1.resample("D")


# In[135]:

df1_day.columns = ["outdoor (daily avg.)"]


# In[136]:

df1_week = df1.resample("7D")


# In[137]:

df1_week.columns = ["outdoor (weekly avg.)"]


# In[138]:

df1_month = df1.resample("M")


# In[139]:

df1_month.columns = ["outdoor (monthly avg.)"]


# In[140]:

df_diff = (df1.resample("D").outdoor - df2.resample("D").indoor)


# In[141]:

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

df1_hour.plot(ax=ax1, alpha=0.25)
df1_day.plot(ax=ax1)
df1_week.plot(ax=ax1)
df1_month.plot(ax=ax1)

df_diff.plot(ax=ax2)
ax2.set_title("temperature difference between outdoor and indoor")

fig.tight_layout()
fig.savefig("ch12-timeseries-resampled.pdf")


# In[142]:

fill_methods = [None, 'ffill', 'bfill']


# In[143]:

pd.concat([df1.resample("5min", fill_method=fm).rename(columns={"outdoor": fm})
           for fm in fill_methods], axis=1).head()


# ## Selected day

# In[144]:

df1_dec25 = df1[(df1.index < "2014-9-1") & (df1.index >= "2014-8-1")].resample("D")


# In[145]:

df1_dec25 = df1.ix["2014-12-25"]


# In[146]:

df1_dec25.head(5)


# In[147]:

df2_dec25 = df2.ix["2014-12-25"]


# In[148]:

df2_dec25.head(5)


# In[149]:

df1_dec25.describe().T


# In[150]:

fig, ax = plt.subplots(1, 1, figsize=(12, 4))

df1_dec25.plot(ax=ax)

fig.savefig("ch12-timeseries-selected-month.pdf")


# In[151]:

df1.index


# # Seaborn statistical visualization library

# In[152]:

import seaborn as sns


# In[153]:

sns.set(style="darkgrid")


# In[154]:

#sns.set(style="whitegrid")


# In[155]:

df1 = pd.read_csv('temperature_outdoor_2014.tsv', delimiter="\t", names=["time", "outdoor"])
df1.time = pd.to_datetime(df1.time.values, unit="s").tz_localize('UTC').tz_convert('Europe/Stockholm')
df1 = df1.set_index("time").resample("10min")
df2 = pd.read_csv('temperature_indoor_2014.tsv', delimiter="\t", names=["time", "indoor"])
df2.time = pd.to_datetime(df2.time.values, unit="s").tz_localize('UTC').tz_convert('Europe/Stockholm')
df2 = df2.set_index("time").resample("10min")
df_temp = pd.concat([df1, df2], axis=1)


# In[156]:

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
df_temp.resample("D").plot(y=["outdoor", "indoor"], ax=ax)
fig.tight_layout()
fig.savefig("ch12-seaborn-plot.pdf")


# In[157]:

#sns.kdeplot(df_temp["outdoor"].dropna().values, shade=True, cumulative=True);


# In[158]:

sns.distplot(df_temp.to_period("M")["outdoor"]["2014-04"].dropna().values, bins=50);
sns.distplot(df_temp.to_period("M")["indoor"]["2014-04"].dropna().values, bins=50);

plt.savefig("ch12-seaborn-distplot.pdf")


# In[159]:

with sns.axes_style("white"):
    sns.jointplot(df_temp.resample("H")["outdoor"].values,
                  df_temp.resample("H")["indoor"].values, kind="hex");
    
plt.savefig("ch12-seaborn-jointplot.pdf")


# In[160]:

sns.kdeplot(df_temp.resample("H")["outdoor"].dropna().values,
            df_temp.resample("H")["indoor"].dropna().values, shade=False);

plt.savefig("ch12-seaborn-kdeplot.pdf")


# In[163]:

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

sns.boxplot(df_temp.dropna(), ax=ax1, palette="pastel")
sns.violinplot(df_temp.dropna(), ax=ax2, palette="pastel")

fig.tight_layout()
fig.savefig("ch12-seaborn-boxplot-violinplot.pdf")


# In[164]:

sns.violinplot(x=df_temp.dropna().index.month, y=df_temp.dropna().outdoor, color="skyblue");

plt.savefig("ch12-seaborn-violinplot.pdf")


# In[165]:

df_temp["month"] = df_temp.index.month
df_temp["hour"] = df_temp.index.hour


# In[166]:

df_temp.head()


# In[167]:

table = pd.pivot_table(df_temp, values='outdoor', index=['month'], columns=['hour'], aggfunc=np.mean)


# In[168]:

table


# In[169]:

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
sns.heatmap(table, ax=ax);

fig.tight_layout()
fig.savefig("ch12-seaborn-heatmap.pdf")


# ## Versions

# In[170]:

get_ipython().magic(u'reload_ext version_information')


# In[171]:

get_ipython().magic(u'version_information numpy, matplotlib, pandas, seaborn')

