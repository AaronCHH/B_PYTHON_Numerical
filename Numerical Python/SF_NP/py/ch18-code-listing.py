
# coding: utf-8

# # Chapter 18: Code listing

# Robert Johansson
# 
# Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).
# 
# The source code listings can be downloaded from http://www.apress.com/9781484205549

# ## Imports

# In[1]:

from __future__ import print_function


# In[2]:

import numpy as np
np.random.seed(0)


# In[3]:

import pandas as pd


# In[4]:

import csv


# In[5]:

import json


# In[6]:

import h5py


# In[7]:

import tables


# In[8]:

import pickle
import cPickle


# In[9]:

import msgpack


# # CSV

# In[10]:

get_ipython().run_cell_magic(u'writefile', u'playerstats-2013-2014.csv', u'# 2013-2014 / Regular Season / All Skaters / Summary / Points\nRank,Player,Team,Pos,GP,G,A,P,+/-,PIM,PPG,PPP,SHG,SHP,GW,OT,S,S%,TOI/GP,Shift/GP,FO%\n1,Sidney Crosby,PIT,C,80,36,68,104,+18,46,11,38,0,0,5,1,259,13.9,21:58,24.0,52.5\n2,Ryan Getzlaf,ANA,C,77,31,56,87,+28,31,5,23,0,0,7,1,204,15.2,21:17,25.2,49.0\n3,Claude Giroux,PHI,C,82,28,58,86,+7,46,7,37,0,0,7,1,223,12.6,20:26,25.1,52.9\n4,Tyler Seguin,DAL,C,80,37,47,84,+16,18,11,25,0,0,8,0,294,12.6,19:20,23.4,41.5\n5,Corey Perry,ANA,R,81,43,39,82,+32,65,8,18,0,0,9,1,280,15.4,19:28,23.2,36.0')


# In[11]:

get_ipython().run_cell_magic(u'writefile', u'playerstats-2013-2014-top30.csv', u'# 2013-2014 / Regular Season / All Skaters / Summary / Points\nRank,Player,Team,Pos,GP,G,A,P,+/-,PIM,PPG,PPP,SHG,SHP,GW,OT,S,S%,TOI/GP,Shift/GP,FO%\n1,Sidney Crosby,PIT,C,80,36,68,104,+18,46,11,38,0,0,5,1,259,13.9,21:58,24.0,52.5\n2,Ryan Getzlaf,ANA,C,77,31,56,87,+28,31,5,23,0,0,7,1,204,15.2,21:17,25.2,49.0\n3,Claude Giroux,PHI,C,82,28,58,86,+7,46,7,37,0,0,7,1,223,12.6,20:26,25.1,52.9\n4,Tyler Seguin,DAL,C,80,37,47,84,+16,18,11,25,0,0,8,0,294,12.6,19:20,23.4,41.5\n5,Corey Perry,ANA,R,81,43,39,82,+32,65,8,18,0,0,9,1,280,15.4,19:28,23.2,36.0\n6,Phil Kessel,TOR,R,82,37,43,80,-5,27,8,20,0,0,6,0,305,12.1,20:39,24.5,14.3\n7,Taylor Hall,EDM,L,75,27,53,80,-15,44,7,17,0,1,1,1,250,10.8,20:00,25.4,45.7\n8,Alex Ovechkin,WSH,L,78,51,28,79,-35,48,24,39,0,1,10,3,386,13.2,20:32,21.8,66.7\n9,Joe Pavelski,SJS,C,82,41,38,79,+23,32,16,31,1,2,3,0,225,18.2,19:51,27.1,56.0\n10,Jamie Benn,DAL,L,81,34,45,79,+21,64,5,19,1,3,3,1,279,12.2,19:09,25.0,52.8\n11,Nicklas Backstrom,WSH,C,82,18,61,79,-20,54,6,44,1,1,1,0,196,9.2,19:48,23.3,50.4\n12,Patrick Sharp,CHI,L,82,34,44,78,+13,40,10,25,0,0,3,1,313,10.9,18:53,22.7,54.6\n13,Joe Thornton,SJS,C,82,11,65,76,+20,32,2,19,0,1,3,1,122,9.0,18:55,26.3,56.1\n14,Erik Karlsson,OTT,D,82,20,54,74,-15,36,5,31,0,0,1,0,257,7.8,27:04,28.6,0.0\n15,Evgeni Malkin,PIT,C,60,23,49,72,+10,62,7,30,0,0,3,0,191,12.0,20:03,21.4,48.8\n16,Patrick Marleau,SJS,L,82,33,37,70,+0,18,11,23,2,2,4,0,285,11.6,20:31,27.3,52.9\n17,Anze Kopitar,LAK,C,82,29,41,70,+34,24,10,23,0,0,9,2,200,14.5,20:53,25.4,53.3\n18,Matt Duchene,COL,C,71,23,47,70,+8,19,5,17,0,0,6,1,217,10.6,18:29,22.0,50.3\n19,Martin St. Louis,"TBL, NYR",R,81,30,39,69,+13,10,9,21,1,2,5,1,204,14.7,20:56,25.7,40.7\n20,Patrick Kane,CHI,R,69,29,40,69,+7,22,10,25,0,0,6,0,227,12.8,19:36,22.9,50.0\n21,Blake Wheeler,WPG,R,82,28,41,69,+4,63,8,19,0,0,4,2,225,12.4,18:41,24.0,37.5\n22,Kyle Okposo,NYI,R,71,27,42,69,-9,51,5,15,0,0,4,1,195,13.8,20:26,22.2,47.5\n23,David Krejci,BOS,C,80,19,50,69,+39,28,3,19,0,0,6,1,169,11.2,19:07,21.3,51.2\n24,Chris Kunitz,PIT,L,78,35,33,68,+25,66,13,22,0,0,8,0,218,16.1,19:09,22.2,75.0\n25,Jonathan Toews,CHI,C,76,28,40,68,+26,34,5,15,3,5,5,0,193,14.5,20:28,25.9,57.2\n26,Thomas Vanek,"BUF, NYI, MTL",L,78,27,41,68,+7,46,8,18,0,0,4,0,248,10.9,19:21,21.6,43.5\n27,Jaromir Jagr,NJD,R,82,24,43,67,+16,46,5,17,0,0,6,1,231,10.4,19:09,22.8,0.0\n28,John Tavares,NYI,C,59,24,42,66,-6,40,8,25,0,0,4,0,188,12.8,21:14,22.3,49.1\n29,Jason Spezza,OTT,C,75,23,43,66,-26,46,9,22,0,0,5,0,223,10.3,18:12,23.8,54.0\n30,Jordan Eberle,EDM,R,80,28,37,65,-11,18,7,20,1,1,4,1,200,14.0,19:32,25.4,38.1')


# In[12]:

get_ipython().system(u'head -n 5 playerstats-2013-2014-top30.csv')


# In[13]:

rows = []


# In[14]:

with open("playerstats-2013-2014.csv") as f:
    csvreader = csv.reader(f, )
    print(type(csvreader))
    for fields in csvreader:
        rows.append(fields)


# In[15]:

rows[1][1:6]


# In[16]:

rows[2][1:6]


# In[17]:

data = np.random.randn(100, 3)


# In[18]:

np.savetxt("data.csv", data, delimiter=",", header="x, y, z", comments="# Random x, y, z coordinates\n")


# In[19]:

get_ipython().system(u'head -n 5 data.csv')


# In[20]:

data_load = np.loadtxt("data.csv", skiprows=2, delimiter=",")


# In[21]:

data_load[1,:]


# In[22]:

data_load.dtype


# In[23]:

(data == data_load).all()


# In[24]:

np.loadtxt("playerstats-2013-2014.csv", skiprows=2, delimiter=",")


# In[25]:

data = np.loadtxt("playerstats-2013-2014.csv", skiprows=2, delimiter=",", dtype=bytes)


# In[26]:

data[0][1:6]


# In[27]:

np.loadtxt("playerstats-2013-2014.csv", skiprows=2, delimiter=",", usecols=[6,7,8])


# In[28]:

df = pd.read_csv("playerstats-2013-2014.csv", skiprows=1)


# In[29]:

df = df.set_index("Rank")


# In[30]:

df[["Player", "GP", "G", "A", "P"]]


# In[31]:

df.info()


# In[32]:

df[["Player", "GP", "G", "A", "P"]].to_csv("playerstats-2013-2014-subset.csv")


# In[33]:

get_ipython().system(u'head -n 5 playerstats-2013-2014-subset.csv')


# #HDF5

# ## h5py

# In[34]:

import h5py


# In[35]:

# mode = "w", "r", "w-", "r+", "a"


# In[36]:

f = h5py.File("data.h5", "w")


# In[37]:

f.mode


# In[38]:

f.flush()


# In[39]:

f.close()


# In[40]:

f = h5py.File("data.h5", "w")


# In[41]:

f.name


# In[42]:

grp1 = f.create_group("experiment1")


# In[43]:

grp1.name


# In[44]:

grp2_meas = f.create_group("experiment2/measurement")


# In[45]:

grp2_meas.name


# In[46]:

grp2_sim = f.create_group("experiment2/simulation")


# In[47]:

grp2_sim.name


# In[48]:

f["/experiment1"]


# In[49]:

f["/experiment2/simulation"]


# In[50]:

grp_expr2 = f["/experiment2"]


# In[51]:

grp_expr2['simulation']


# In[52]:

list(f.keys())


# In[53]:

list(f.items())


# In[54]:

f.visit(lambda x: print(x))


# In[55]:

f.visititems(lambda name, value: print(name, value))


# In[56]:

"experiment1" in f


# In[57]:

"simulation" in f["experiment2"]


# In[58]:

"experiment3" in f


# In[59]:

f.flush()


# In[60]:

get_ipython().system(u'h5ls -r data.h5')


# In[61]:

data1 = np.arange(10)


# In[62]:

data2 = np.random.randn(100, 100)


# In[63]:

f["array1"] = data1


# In[64]:

f["/experiment2/measurement/meas1"] = data2


# In[65]:

f.visititems(lambda name, value: print(name, value))


# In[66]:

ds = f["array1"]


# In[67]:

ds


# In[68]:

ds.name


# In[69]:

ds.dtype


# In[70]:

ds.shape


# In[71]:

ds.len()


# In[72]:

ds.value


# In[73]:

ds = f["/experiment2/measurement/meas1"]


# In[74]:

ds


# In[75]:

ds.dtype


# In[76]:

ds.shape


# In[77]:

data_full = ds[...]


# In[78]:

type(data_full)


# In[79]:

data_full.shape


# In[80]:

data_col = ds[:, 0]


# In[81]:

data_col.shape


# In[82]:

ds[10:20:3, 10:20:3]


# In[83]:

ds[[1,2,3], :].shape


# In[84]:

ds[[1,2,3], :].shape


# In[85]:

mask = ds[:, 0] > 2.0


# In[86]:

mask.shape, mask.dtype


# In[87]:

ds[mask, 0]


# In[88]:

ds[mask, :5]


# In[89]:

# create empty data sets, assign and update datasets


# In[90]:

ds = f.create_dataset("array2", data=np.random.randint(10, size=10))


# In[91]:

ds


# In[92]:

ds.value


# In[93]:

ds = f.create_dataset("/experiment2/simulation/data1", shape=(5, 5), fillvalue=-1)


# In[94]:

ds


# In[95]:

ds.value


# In[96]:

ds = f.create_dataset("/experiment1/simulation/data1", shape=(5000, 5000, 5000),
                      fillvalue=0, compression='gzip')


# In[97]:

ds


# In[98]:

ds[:, 0, 0] = np.random.rand(5000)


# In[99]:

ds[1, :, 0] += np.random.rand(5000)


# In[100]:

ds[:2, :5, 0]


# In[101]:

ds.fillvalue


# In[102]:

f["experiment1"].visititems(lambda name, value: print(name, value))


# In[103]:

float(np.prod(ds.shape) * ds[0,0,0].nbytes) / (1024**3)  # Gb


# In[104]:

f.flush()


# In[105]:

f.filename


# In[106]:

get_ipython().system(u'ls -lh data.h5')


# In[107]:

del f["/experiment1/simulation/data1"]


# In[108]:

f["experiment1"].visititems(lambda name, value: print(name, value))


# In[109]:

f.close()


# In[110]:

# attributes


# In[111]:

f = h5py.File("data.h5")


# In[112]:

f.attrs


# In[113]:

f.attrs["desc"] = "Result sets from experiments and simulations"


# In[114]:

f["experiment1"].attrs["date"] = "2015-1-1"


# In[115]:

f["experiment2"].attrs["date"] = "2015-1-2"


# In[116]:

f["experiment2/simulation/data1"].attrs["k"] = 1.5


# In[117]:

f["experiment2/simulation/data1"].attrs["T"] = 1000


# In[118]:

list(f["experiment1"].attrs.keys())


# In[119]:

list(f["experiment2/simulation/data1"].attrs.items())


# In[120]:

"T" in f["experiment2/simulation/data1"].attrs


# In[121]:

del f["experiment2/simulation/data1"].attrs["T"]


# In[122]:

"T" in f["experiment2/simulation/data1"].attrs


# In[123]:

f["experiment2/simulation/data1"].attrs["t"] = np.array([1, 2, 3])


# In[124]:

f["experiment2/simulation/data1"].attrs["t"]


# In[125]:

f.close()


# ## pytables

# In[126]:

df = pd.read_csv("playerstats-2013-2014-top30.csv", skiprows=1)
df = df.set_index("Rank")


# In[127]:

df[["Player", "Pos", "GP", "P", "G", "A", "S%", "Shift/GP"]].head(5)


# In[128]:

f = tables.open_file("playerstats-2013-2014.h5", mode="w")


# In[129]:

grp = f.create_group("/", "season_2013_2014", title="NHL player statistics for the 2013/2014 season")


# In[130]:

grp


# In[131]:

f.root


# In[132]:

class PlayerStat(tables.IsDescription):
    player = tables.StringCol(20, dflt="")
    position = tables.StringCol(1, dflt="C")
    games_played = tables.UInt8Col(dflt=0)
    points = tables.UInt16Col(dflt=0)
    goals = tables.UInt16Col(dflt=0)
    assists = tables.UInt16Col(dflt=0)
    shooting_percentage = tables.Float64Col(dflt=0.0)
    shifts_per_game_played = tables.Float64Col(dflt=0.0) 


# In[133]:

top30_table = f.create_table(grp, 'top30', PlayerStat, "Top 30 point leaders")


# In[134]:

playerstat = top30_table.row


# In[135]:

type(playerstat)


# In[136]:

for index, row_series in df.iterrows():
    playerstat["player"] = row_series["Player"]    
    playerstat["position"] = row_series["Pos"]    
    playerstat["games_played"] = row_series["GP"]    
    playerstat["points"] = row_series["P"]    
    playerstat["goals"] = row_series["G"]
    playerstat["assists"] = row_series["A"] 
    playerstat["shooting_percentage"] = row_series["S%"]
    playerstat["shifts_per_game_played"] = row_series["Shift/GP"]
    playerstat.append()


# In[137]:

top30_table.flush()


# In[138]:

top30_table.cols.player[:5]


# In[139]:

top30_table.cols.points[:5]


# In[140]:

def print_playerstat(row):
    print("%20s\t%s\t%s\t%s" %
          (row["player"].decode('UTF-8'), row["points"], row["goals"], row["assists"]))


# In[141]:

for row in top30_table.iterrows():
    print_playerstat(row)


# In[142]:

for row in top30_table.where("(points > 75) & (points <= 80)"):
    print_playerstat(row)


# In[143]:

for row in top30_table.where("(goals > 40) & (points < 80)"):
    print_playerstat(row)


# In[144]:

f


# In[145]:

f.flush()


# In[146]:

f.close()


# In[147]:

get_ipython().system(u'h5ls -rv playerstats-2013-2014.h5')


# ## Pandas hdfstore

# In[148]:

import pandas as pd


# In[149]:

store = pd.HDFStore('store.h5')


# In[150]:

df = pd.DataFrame(np.random.rand(5,5))


# In[151]:

store["df1"] = df


# In[152]:

df = pd.read_csv("playerstats-2013-2014-top30.csv", skiprows=1)


# In[153]:

store["df2"] = df


# In[154]:

store.keys()


# In[155]:

'df2' in store


# In[156]:

df = store["df1"]


# In[157]:

store.root


# In[158]:

store.close()


# In[159]:

f = h5py.File("store.h5")


# In[160]:

f.visititems(lambda x, y: print(x, "\t" * int(3 - len(str(x))//8), y))


# In[161]:

f["/df2/block0_items"].value          


# In[162]:

f["/df2/block0_values"][:3]


# In[163]:

f["/df2/block1_items"].value  


# In[164]:

f["/df2/block1_values"][:3, :5]


# # JSON

# In[165]:

data = ["string", 1.0, 2, None]


# In[166]:

data_json = json.dumps(data)


# In[167]:

data_json


# In[168]:

data2 = json.loads(data_json)


# In[169]:

data


# In[170]:

data[0]


# In[171]:

data = {"one": 1, "two": 2.0, "three": "three"}


# In[172]:

data_json = json.dumps(data)


# In[173]:

print(data_json)


# In[174]:

data = json.loads(data_json)


# In[175]:

data["two"]


# In[176]:

data["three"]


# In[177]:

data = {"one": [1], 
        "two": [1, 2], 
        "three": [1, 2, 3]}


# In[178]:

data_json = json.dumps(data, indent=True)


# In[179]:

print(data_json)


# In[180]:

data = {"one": [1], 
        "two": {"one": 1, "two": 2}, 
        "three": [(1,), (1, 2), (1, 2, 3)],
        "four": "a text string"}


# In[181]:

with open("data.json", "w") as f:
    json.dump(data, f)


# In[182]:

get_ipython().system(u'cat data.json')


# In[183]:

with open("data.json", "r") as f:
    data_from_file = json.load(f)


# In[184]:

data_from_file["two"]


# In[185]:

data_from_file["three"]


# In[186]:

get_ipython().system(u'head -n 20 tokyo-metro.json')


# In[187]:

get_ipython().system(u'wc tokyo-metro.json')


# In[188]:

with open("tokyo-metro.json", "r") as f:
    data = json.load(f)


# In[189]:

data.keys()


# In[190]:

data["C"].keys()


# In[191]:

data["C"]["color"]


# In[192]:

data["C"]["transfers"]


# In[193]:

[(s, e, tt) for s, e, tt in data["C"]["travel_times"] if tt == 1]


# In[194]:

data


# In[195]:

get_ipython().system(u'ls -lh tokyo-metro.json')


# In[196]:

data_pack = msgpack.packb(data)


# In[197]:

del data


# In[198]:

type(data_pack)


# In[199]:

len(data_pack)


# In[200]:

with open("tokyo-metro.msgpack", "wb") as f:
    f.write(data_pack)


# In[201]:

get_ipython().system(u'ls -lh tokyo-metro.msgpack')


# In[202]:

with open("tokyo-metro.msgpack", "rb") as f:
    data_msgpack = f.read()
    data = msgpack.unpackb(data_msgpack)


# In[203]:

list(data.keys())


# In[204]:

with open("tokyo-metro.pickle", "wb") as f:
    cPickle.dump(data, f)


# In[205]:

del data


# In[206]:

get_ipython().system(u'ls -lh tokyo-metro.pickle')


# In[207]:

with open("tokyo-metro.pickle", "rb") as f:
    data = pickle.load(f)


# In[208]:

data.keys()


# # Versions

# In[209]:

get_ipython().magic(u'reload_ext version_information')


# In[210]:

get_ipython().magic(u'version_information numpy, pandas, csv, json, tables, h5py, msgpack')

