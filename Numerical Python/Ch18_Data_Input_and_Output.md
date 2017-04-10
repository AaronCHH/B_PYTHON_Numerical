
# Chapter 18: Code listing
<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Chapter 18: Code listing](#chapter-18-code-listing)
  * [Imports](#imports)
* [CSV](#csv)
  * [h5py](#h5py)
  * [pytables](#pytables)
  * [Pandas hdfstore](#pandas-hdfstore)
* [JSON](#json)
* [Versions](#versions)

<!-- tocstop -->


---

Robert Johansson

Source code listings for [Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) (ISBN 978-1-484205-54-9).

The source code listings can be downloaded from http://www.apress.com/9781484205549

## Imports


```python
from __future__ import print_function
```


```python
import numpy as np
np.random.seed(0)
```


```python
import pandas as pd
```


```python
import csv
```


```python
import json
```


```python
import h5py
```


```python
import tables
```


```python
import pickle
import cPickle
```


```python
import msgpack
```

# CSV


```python
%%writefile playerstats-2013-2014.csv
# 2013-2014 / Regular Season / All Skaters / Summary / Points
Rank,Player,Team,Pos,GP,G,A,P,+/-,PIM,PPG,PPP,SHG,SHP,GW,OT,S,S%,TOI/GP,Shift/GP,FO%
1,Sidney Crosby,PIT,C,80,36,68,104,+18,46,11,38,0,0,5,1,259,13.9,21:58,24.0,52.5
2,Ryan Getzlaf,ANA,C,77,31,56,87,+28,31,5,23,0,0,7,1,204,15.2,21:17,25.2,49.0
3,Claude Giroux,PHI,C,82,28,58,86,+7,46,7,37,0,0,7,1,223,12.6,20:26,25.1,52.9
4,Tyler Seguin,DAL,C,80,37,47,84,+16,18,11,25,0,0,8,0,294,12.6,19:20,23.4,41.5
5,Corey Perry,ANA,R,81,43,39,82,+32,65,8,18,0,0,9,1,280,15.4,19:28,23.2,36.0
```

    Overwriting playerstats-2013-2014.csv



```python
%%writefile playerstats-2013-2014-top30.csv
# 2013-2014 / Regular Season / All Skaters / Summary / Points
Rank,Player,Team,Pos,GP,G,A,P,+/-,PIM,PPG,PPP,SHG,SHP,GW,OT,S,S%,TOI/GP,Shift/GP,FO%
1,Sidney Crosby,PIT,C,80,36,68,104,+18,46,11,38,0,0,5,1,259,13.9,21:58,24.0,52.5
2,Ryan Getzlaf,ANA,C,77,31,56,87,+28,31,5,23,0,0,7,1,204,15.2,21:17,25.2,49.0
3,Claude Giroux,PHI,C,82,28,58,86,+7,46,7,37,0,0,7,1,223,12.6,20:26,25.1,52.9
4,Tyler Seguin,DAL,C,80,37,47,84,+16,18,11,25,0,0,8,0,294,12.6,19:20,23.4,41.5
5,Corey Perry,ANA,R,81,43,39,82,+32,65,8,18,0,0,9,1,280,15.4,19:28,23.2,36.0
6,Phil Kessel,TOR,R,82,37,43,80,-5,27,8,20,0,0,6,0,305,12.1,20:39,24.5,14.3
7,Taylor Hall,EDM,L,75,27,53,80,-15,44,7,17,0,1,1,1,250,10.8,20:00,25.4,45.7
8,Alex Ovechkin,WSH,L,78,51,28,79,-35,48,24,39,0,1,10,3,386,13.2,20:32,21.8,66.7
9,Joe Pavelski,SJS,C,82,41,38,79,+23,32,16,31,1,2,3,0,225,18.2,19:51,27.1,56.0
10,Jamie Benn,DAL,L,81,34,45,79,+21,64,5,19,1,3,3,1,279,12.2,19:09,25.0,52.8
11,Nicklas Backstrom,WSH,C,82,18,61,79,-20,54,6,44,1,1,1,0,196,9.2,19:48,23.3,50.4
12,Patrick Sharp,CHI,L,82,34,44,78,+13,40,10,25,0,0,3,1,313,10.9,18:53,22.7,54.6
13,Joe Thornton,SJS,C,82,11,65,76,+20,32,2,19,0,1,3,1,122,9.0,18:55,26.3,56.1
14,Erik Karlsson,OTT,D,82,20,54,74,-15,36,5,31,0,0,1,0,257,7.8,27:04,28.6,0.0
15,Evgeni Malkin,PIT,C,60,23,49,72,+10,62,7,30,0,0,3,0,191,12.0,20:03,21.4,48.8
16,Patrick Marleau,SJS,L,82,33,37,70,+0,18,11,23,2,2,4,0,285,11.6,20:31,27.3,52.9
17,Anze Kopitar,LAK,C,82,29,41,70,+34,24,10,23,0,0,9,2,200,14.5,20:53,25.4,53.3
18,Matt Duchene,COL,C,71,23,47,70,+8,19,5,17,0,0,6,1,217,10.6,18:29,22.0,50.3
19,Martin St. Louis,"TBL, NYR",R,81,30,39,69,+13,10,9,21,1,2,5,1,204,14.7,20:56,25.7,40.7
20,Patrick Kane,CHI,R,69,29,40,69,+7,22,10,25,0,0,6,0,227,12.8,19:36,22.9,50.0
21,Blake Wheeler,WPG,R,82,28,41,69,+4,63,8,19,0,0,4,2,225,12.4,18:41,24.0,37.5
22,Kyle Okposo,NYI,R,71,27,42,69,-9,51,5,15,0,0,4,1,195,13.8,20:26,22.2,47.5
23,David Krejci,BOS,C,80,19,50,69,+39,28,3,19,0,0,6,1,169,11.2,19:07,21.3,51.2
24,Chris Kunitz,PIT,L,78,35,33,68,+25,66,13,22,0,0,8,0,218,16.1,19:09,22.2,75.0
25,Jonathan Toews,CHI,C,76,28,40,68,+26,34,5,15,3,5,5,0,193,14.5,20:28,25.9,57.2
26,Thomas Vanek,"BUF, NYI, MTL",L,78,27,41,68,+7,46,8,18,0,0,4,0,248,10.9,19:21,21.6,43.5
27,Jaromir Jagr,NJD,R,82,24,43,67,+16,46,5,17,0,0,6,1,231,10.4,19:09,22.8,0.0
28,John Tavares,NYI,C,59,24,42,66,-6,40,8,25,0,0,4,0,188,12.8,21:14,22.3,49.1
29,Jason Spezza,OTT,C,75,23,43,66,-26,46,9,22,0,0,5,0,223,10.3,18:12,23.8,54.0
30,Jordan Eberle,EDM,R,80,28,37,65,-11,18,7,20,1,1,4,1,200,14.0,19:32,25.4,38.1
```

    Overwriting playerstats-2013-2014-top30.csv



```python
!head -n 5 playerstats-2013-2014-top30.csv
```

    # 2013-2014 / Regular Season / All Skaters / Summary / Points

    Rank,Player,Team,Pos,GP,G,A,P,+/-,PIM,PPG,PPP,SHG,SHP,GW,OT,S,S%,TOI/GP,Shift/GP,FO%

    1,Sidney Crosby,PIT,C,80,36,68,104,+18,46,11,38,0,0,5,1,259,13.9,21:58,24.0,52.5

    2,Ryan Getzlaf,ANA,C,77,31,56,87,+28,31,5,23,0,0,7,1,204,15.2,21:17,25.2,49.0

    3,Claude Giroux,PHI,C,82,28,58,86,+7,46,7,37,0,0,7,1,223,12.6,20:26,25.1,52.9




```python
rows = []
```


```python
with open("playerstats-2013-2014.csv") as f:
    csvreader = csv.reader(f, )
    print(type(csvreader))
    for fields in csvreader:
        rows.append(fields)
```

    <type '_csv.reader'>



```python
rows[1][1:6]
```




    ['Player', 'Team', 'Pos', 'GP', 'G']




```python
rows[2][1:6]
```




    ['Sidney Crosby', 'PIT', 'C', '80', '36']




```python
data = np.random.randn(100, 3)
```


```python
np.savetxt("data.csv", data, delimiter=",", header="x, y, z", comments="# Random x, y, z coordinates\n")
```


```python
!head -n 5 data.csv
```

    # Random x, y, z coordinates

    x, y, z

    1.764052345967664026e+00,4.001572083672232938e-01,9.787379841057392005e-01

    2.240893199201457797e+00,1.867557990149967484e+00,-9.772778798764110153e-01

    9.500884175255893682e-01,-1.513572082976978872e-01,-1.032188517935578448e-01




```python
data_load = np.loadtxt("data.csv", skiprows=2, delimiter=",")
```


```python
data_load[1,:]
```




    array([ 2.2408932 ,  1.86755799, -0.97727788])




```python
data_load.dtype
```




    dtype('float64')




```python
(data == data_load).all()
```




    True




```python
np.loadtxt("playerstats-2013-2014.csv", skiprows=2, delimiter=",")
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-24-ae638a27585f> in <module>()
    ----> 1 np.loadtxt("playerstats-2013-2014.csv", skiprows=2, delimiter=",")


    /Users/rob/miniconda/envs/py27-npm/lib/python2.7/site-packages/numpy/lib/npyio.pyc in loadtxt(fname, dtype, comments, delimiter, converters, skiprows, usecols, unpack, ndmin)
        858
        859             # Convert each value according to its column and store
    --> 860             items = [conv(val) for (conv, val) in zip(converters, vals)]
        861             # Then pack it according to the dtype's nesting
        862             items = pack_items(items, packing)


    ValueError: could not convert string to float: Sidney Crosby



```python
data = np.loadtxt("playerstats-2013-2014.csv", skiprows=2, delimiter=",", dtype=bytes)
```


```python
data[0][1:6]
```




    array(['Sidney Crosby', 'PIT', 'C', '80', '36'],
          dtype='|S13')




```python
np.loadtxt("playerstats-2013-2014.csv", skiprows=2, delimiter=",", usecols=[6,7,8])
```




    array([[  68.,  104.,   18.],
           [  56.,   87.,   28.],
           [  58.,   86.,    7.],
           [  47.,   84.,   16.],
           [  39.,   82.,   32.]])




```python
df = pd.read_csv("playerstats-2013-2014.csv", skiprows=1)
```


```python
df = df.set_index("Rank")
```


```python
df[["Player", "GP", "G", "A", "P"]]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>GP</th>
      <th>G</th>
      <th>A</th>
      <th>P</th>
    </tr>
    <tr>
      <th>Rank</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Sidney Crosby</td>
      <td>80</td>
      <td>36</td>
      <td>68</td>
      <td>104</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ryan Getzlaf</td>
      <td>77</td>
      <td>31</td>
      <td>56</td>
      <td>87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Claude Giroux</td>
      <td>82</td>
      <td>28</td>
      <td>58</td>
      <td>86</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tyler Seguin</td>
      <td>80</td>
      <td>37</td>
      <td>47</td>
      <td>84</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Corey Perry</td>
      <td>81</td>
      <td>43</td>
      <td>39</td>
      <td>82</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 5 entries, 1 to 5
    Data columns (total 20 columns):
    Player      5 non-null object
    Team        5 non-null object
    Pos         5 non-null object
    GP          5 non-null int64
    G           5 non-null int64
    A           5 non-null int64
    P           5 non-null int64
    +/-         5 non-null int64
    PIM         5 non-null int64
    PPG         5 non-null int64
    PPP         5 non-null int64
    SHG         5 non-null int64
    SHP         5 non-null int64
    GW          5 non-null int64
    OT          5 non-null int64
    S           5 non-null int64
    S%          5 non-null float64
    TOI/GP      5 non-null object
    Shift/GP    5 non-null float64
    FO%         5 non-null float64
    dtypes: float64(3), int64(13), object(4)
    memory usage: 840.0+ bytes



```python
df[["Player", "GP", "G", "A", "P"]].to_csv("playerstats-2013-2014-subset.csv")
```


```python
!head -n 5 playerstats-2013-2014-subset.csv
```

    Rank,Player,GP,G,A,P

    1,Sidney Crosby,80,36,68,104

    2,Ryan Getzlaf,77,31,56,87

    3,Claude Giroux,82,28,58,86

    4,Tyler Seguin,80,37,47,84



#HDF5

## h5py


```python
import h5py
```


```python
# mode = "w", "r", "w-", "r+", "a"
```


```python
f = h5py.File("data.h5", "w")
```


```python
f.mode
```




    'r+'




```python
f.flush()
```


```python
f.close()
```


```python
f = h5py.File("data.h5", "w")
```


```python
f.name
```




    u'/'




```python
grp1 = f.create_group("experiment1")
```


```python
grp1.name
```




    u'/experiment1'




```python
grp2_meas = f.create_group("experiment2/measurement")
```


```python
grp2_meas.name
```




    u'/experiment2/measurement'




```python
grp2_sim = f.create_group("experiment2/simulation")
```


```python
grp2_sim.name
```




    u'/experiment2/simulation'




```python
f["/experiment1"]
```




    <HDF5 group "/experiment1" (0 members)>




```python
f["/experiment2/simulation"]
```




    <HDF5 group "/experiment2/simulation" (0 members)>




```python
grp_expr2 = f["/experiment2"]
```


```python
grp_expr2['simulation']
```




    <HDF5 group "/experiment2/simulation" (0 members)>




```python
list(f.keys())
```




    [u'experiment1', u'experiment2']




```python
list(f.items())
```




    [(u'experiment1', <HDF5 group "/experiment1" (0 members)>),
     (u'experiment2', <HDF5 group "/experiment2" (2 members)>)]




```python
f.visit(lambda x: print(x))
```

    experiment1
    experiment2
    experiment2/measurement
    experiment2/simulation



```python
f.visititems(lambda name, value: print(name, value))
```

    experiment1 <HDF5 group "/experiment1" (0 members)>
    experiment2 <HDF5 group "/experiment2" (2 members)>
    experiment2/measurement <HDF5 group "/experiment2/measurement" (0 members)>
    experiment2/simulation <HDF5 group "/experiment2/simulation" (0 members)>



```python
"experiment1" in f
```




    True




```python
"simulation" in f["experiment2"]
```




    True




```python
"experiment3" in f
```




    False




```python
f.flush()
```


```python
!h5ls -r data.h5
```

    /                        Group

    /experiment1             Group

    /experiment2             Group

    /experiment2/measurement Group

    /experiment2/simulation  Group




```python
data1 = np.arange(10)
```


```python
data2 = np.random.randn(100, 100)
```


```python
f["array1"] = data1
```


```python
f["/experiment2/measurement/meas1"] = data2
```


```python
f.visititems(lambda name, value: print(name, value))
```

    array1 <HDF5 dataset "array1": shape (10,), type "<i8">
    experiment1 <HDF5 group "/experiment1" (0 members)>
    experiment2 <HDF5 group "/experiment2" (2 members)>
    experiment2/measurement <HDF5 group "/experiment2/measurement" (1 members)>
    experiment2/measurement/meas1 <HDF5 dataset "meas1": shape (100, 100), type "<f8">
    experiment2/simulation <HDF5 group "/experiment2/simulation" (0 members)>



```python
ds = f["array1"]
```


```python
ds
```




    <HDF5 dataset "array1": shape (10,), type "<i8">




```python
ds.name
```




    u'/array1'




```python
ds.dtype
```




    dtype('int64')




```python
ds.shape
```




    (10,)




```python
ds.len()
```




    10




```python
ds.value
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
ds = f["/experiment2/measurement/meas1"]
```


```python
ds
```




    <HDF5 dataset "meas1": shape (100, 100), type "<f8">




```python
ds.dtype
```




    dtype('float64')




```python
ds.shape
```




    (100, 100)




```python
data_full = ds[...]
```


```python
type(data_full)
```




    numpy.ndarray




```python
data_full.shape
```




    (100, 100)




```python
data_col = ds[:, 0]
```


```python
data_col.shape
```




    (100,)




```python
ds[10:20:3, 10:20:3]
```




    array([[ 0.60270766, -0.34804638, -0.813596  , -1.29737966],
           [ 0.91320192, -1.06343294,  0.22734595,  0.52759738],
           [ 1.25774422, -0.32775492,  1.4849256 ,  0.28005786],
           [-0.84907287, -0.30000358,  1.79691852, -0.19871506]])




```python
ds[[1,2,3], :].shape
```




    (3, 100)




```python
ds[[1,2,3], :].shape
```




    (3, 100)




```python
mask = ds[:, 0] > 2.0
```


```python
mask.shape, mask.dtype
```




    ((100,), dtype('bool'))




```python
ds[mask, 0]
```




    array([ 2.04253623,  2.1041854 ,  2.05689385])




```python
ds[mask, :5]
```




    array([[ 2.04253623, -0.91946118,  0.11467003, -0.1374237 ,  1.36552692],
           [ 2.1041854 ,  0.22725706, -1.1291663 , -0.28133197, -0.7394167 ],
           [ 2.05689385,  0.18041971, -0.06670925, -0.02835398,  0.48480475]])




```python
# create empty data sets, assign and update datasets
```


```python
ds = f.create_dataset("array2", data=np.random.randint(10, size=10))
```


```python
ds
```




    <HDF5 dataset "array2": shape (10,), type "<i8">




```python
ds.value
```




    array([0, 2, 2, 4, 7, 3, 7, 2, 4, 1])




```python
ds = f.create_dataset("/experiment2/simulation/data1", shape=(5, 5), fillvalue=-1)
```


```python
ds
```




    <HDF5 dataset "data1": shape (5, 5), type "<f4">




```python
ds.value
```




    array([[-1., -1., -1., -1., -1.],
           [-1., -1., -1., -1., -1.],
           [-1., -1., -1., -1., -1.],
           [-1., -1., -1., -1., -1.],
           [-1., -1., -1., -1., -1.]], dtype=float32)




```python
ds = f.create_dataset("/experiment1/simulation/data1", shape=(5000, 5000, 5000),
                      fillvalue=0, compression='gzip')
```


```python
ds
```




    <HDF5 dataset "data1": shape (5000, 5000, 5000), type "<f4">




```python
ds[:, 0, 0] = np.random.rand(5000)
```


```python
ds[1, :, 0] += np.random.rand(5000)
```


```python
ds[:2, :5, 0]
```




    array([[ 0.69393438,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 1.4819994 ,  0.01639538,  0.54387355,  0.11130908,  0.99287713]], dtype=float32)




```python
ds.fillvalue
```




    0.0




```python
f["experiment1"].visititems(lambda name, value: print(name, value))
```

    simulation <HDF5 group "/experiment1/simulation" (1 members)>
    simulation/data1 <HDF5 dataset "data1": shape (5000, 5000, 5000), type "<f4">



```python
float(np.prod(ds.shape) * ds[0,0,0].nbytes) / (1024**3)  # Gb
```




    465.66128730773926




```python
f.flush()
```


```python
f.filename
```




    u'data.h5'




```python
!ls -lh data.h5
```

    -rw-r--r--@ 1 rob  staff   357K Aug  3 23:46 data.h5




```python
del f["/experiment1/simulation/data1"]
```


```python
f["experiment1"].visititems(lambda name, value: print(name, value))
```

    simulation <HDF5 group "/experiment1/simulation" (0 members)>



```python
f.close()
```


```python
# attributes
```


```python
f = h5py.File("data.h5")
```


```python
f.attrs
```




    <Attributes of HDF5 object at 4456000960>




```python
f.attrs["desc"] = "Result sets from experiments and simulations"
```


```python
f["experiment1"].attrs["date"] = "2015-1-1"
```


```python
f["experiment2"].attrs["date"] = "2015-1-2"
```


```python
f["experiment2/simulation/data1"].attrs["k"] = 1.5
```


```python
f["experiment2/simulation/data1"].attrs["T"] = 1000
```


```python
list(f["experiment1"].attrs.keys())
```




    [u'date']




```python
list(f["experiment2/simulation/data1"].attrs.items())
```




    [(u'k', 1.5), (u'T', 1000)]




```python
"T" in f["experiment2/simulation/data1"].attrs
```




    True




```python
del f["experiment2/simulation/data1"].attrs["T"]
```


```python
"T" in f["experiment2/simulation/data1"].attrs
```




    False




```python
f["experiment2/simulation/data1"].attrs["t"] = np.array([1, 2, 3])
```


```python
f["experiment2/simulation/data1"].attrs["t"]
```




    array([1, 2, 3])




```python
f.close()
```

## pytables


```python
df = pd.read_csv("playerstats-2013-2014-top30.csv", skiprows=1)
df = df.set_index("Rank")
```


```python
df[["Player", "Pos", "GP", "P", "G", "A", "S%", "Shift/GP"]].head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>Pos</th>
      <th>GP</th>
      <th>P</th>
      <th>G</th>
      <th>A</th>
      <th>S%</th>
      <th>Shift/GP</th>
    </tr>
    <tr>
      <th>Rank</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Sidney Crosby</td>
      <td>C</td>
      <td>80</td>
      <td>104</td>
      <td>36</td>
      <td>68</td>
      <td>13.9</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ryan Getzlaf</td>
      <td>C</td>
      <td>77</td>
      <td>87</td>
      <td>31</td>
      <td>56</td>
      <td>15.2</td>
      <td>25.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Claude Giroux</td>
      <td>C</td>
      <td>82</td>
      <td>86</td>
      <td>28</td>
      <td>58</td>
      <td>12.6</td>
      <td>25.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Tyler Seguin</td>
      <td>C</td>
      <td>80</td>
      <td>84</td>
      <td>37</td>
      <td>47</td>
      <td>12.6</td>
      <td>23.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Corey Perry</td>
      <td>R</td>
      <td>81</td>
      <td>82</td>
      <td>43</td>
      <td>39</td>
      <td>15.4</td>
      <td>23.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
f = tables.open_file("playerstats-2013-2014.h5", mode="w")
```


```python
grp = f.create_group("/", "season_2013_2014", title="NHL player statistics for the 2013/2014 season")
```


```python
grp
```




    /season_2013_2014 (Group) 'NHL player statistics for the 2013/2014 season'
      children := []




```python
f.root
```




    / (RootGroup) ''
      children := ['season_2013_2014' (Group)]




```python
class PlayerStat(tables.IsDescription):
    player = tables.StringCol(20, dflt="")
    position = tables.StringCol(1, dflt="C")
    games_played = tables.UInt8Col(dflt=0)
    points = tables.UInt16Col(dflt=0)
    goals = tables.UInt16Col(dflt=0)
    assists = tables.UInt16Col(dflt=0)
    shooting_percentage = tables.Float64Col(dflt=0.0)
    shifts_per_game_played = tables.Float64Col(dflt=0.0)
```


```python
top30_table = f.create_table(grp, 'top30', PlayerStat, "Top 30 point leaders")
```


```python
playerstat = top30_table.row
```


```python
type(playerstat)
```




    tables.tableextension.Row




```python
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
```


```python
top30_table.flush()
```


```python
top30_table.cols.player[:5]
```




    array(['Sidney Crosby', 'Ryan Getzlaf', 'Claude Giroux', 'Tyler Seguin',
           'Corey Perry'],
          dtype='|S20')




```python
top30_table.cols.points[:5]
```




    array([104,  87,  86,  84,  82], dtype=uint16)




```python
def print_playerstat(row):
    print("%20s\t%s\t%s\t%s" %
          (row["player"].decode('UTF-8'), row["points"], row["goals"], row["assists"]))
```


```python
for row in top30_table.iterrows():
    print_playerstat(row)
```

           Sidney Crosby	104	36	68
            Ryan Getzlaf	87	31	56
           Claude Giroux	86	28	58
            Tyler Seguin	84	37	47
             Corey Perry	82	43	39
             Phil Kessel	80	37	43
             Taylor Hall	80	27	53
           Alex Ovechkin	79	51	28
            Joe Pavelski	79	41	38
              Jamie Benn	79	34	45
       Nicklas Backstrom	79	18	61
           Patrick Sharp	78	34	44
            Joe Thornton	76	11	65
           Erik Karlsson	74	20	54
           Evgeni Malkin	72	23	49
         Patrick Marleau	70	33	37
            Anze Kopitar	70	29	41
            Matt Duchene	70	23	47
        Martin St. Louis	69	30	39
            Patrick Kane	69	29	40
           Blake Wheeler	69	28	41
             Kyle Okposo	69	27	42
            David Krejci	69	19	50
            Chris Kunitz	68	35	33
          Jonathan Toews	68	28	40
            Thomas Vanek	68	27	41
            Jaromir Jagr	67	24	43
            John Tavares	66	24	42
            Jason Spezza	66	23	43
           Jordan Eberle	65	28	37



```python
for row in top30_table.where("(points > 75) & (points <= 80)"):
    print_playerstat(row)
```

             Phil Kessel	80	37	43
             Taylor Hall	80	27	53
           Alex Ovechkin	79	51	28
            Joe Pavelski	79	41	38
              Jamie Benn	79	34	45
       Nicklas Backstrom	79	18	61
           Patrick Sharp	78	34	44
            Joe Thornton	76	11	65



```python
for row in top30_table.where("(goals > 40) & (points < 80)"):
    print_playerstat(row)
```

           Alex Ovechkin	79	51	28
            Joe Pavelski	79	41	38



```python
f
```




    File(filename=playerstats-2013-2014.h5, title='', mode='w', root_uep='/', filters=Filters(complevel=0, shuffle=False, fletcher32=False, least_significant_digit=None))
    / (RootGroup) ''
    /season_2013_2014 (Group) 'NHL player statistics for the 2013/2014 season'
    /season_2013_2014/top30 (Table(30,)) 'Top 30 point leaders'
      description := {
      "assists": UInt16Col(shape=(), dflt=0, pos=0),
      "games_played": UInt8Col(shape=(), dflt=0, pos=1),
      "goals": UInt16Col(shape=(), dflt=0, pos=2),
      "player": StringCol(itemsize=20, shape=(), dflt='', pos=3),
      "points": UInt16Col(shape=(), dflt=0, pos=4),
      "position": StringCol(itemsize=1, shape=(), dflt='C', pos=5),
      "shifts_per_game_played": Float64Col(shape=(), dflt=0.0, pos=6),
      "shooting_percentage": Float64Col(shape=(), dflt=0.0, pos=7)}
      byteorder := 'little'
      chunkshape := (1489,)




```python
f.flush()
```


```python
f.close()
```


```python
!h5ls -rv playerstats-2013-2014.h5
```

    Opened "playerstats-2013-2014.h5" with sec2 driver.

    /                        Group

        Attribute: CLASS scalar

            Type:      5-byte null-terminated ASCII string

            Data:  "GROUP"

        Attribute: PYTABLES_FORMAT_VERSION scalar

            Type:      3-byte null-terminated ASCII string

            Data:  "2.1"

        Attribute: TITLE scalar

            Type:      1-byte null-terminated ASCII string

            Data:  ""

        Attribute: VERSION scalar

            Type:      3-byte null-terminated ASCII string

            Data:  "1.0"

        Location:  1:96

        Links:     1

    /season_2013_2014        Group

        Attribute: CLASS scalar

            Type:      5-byte null-terminated ASCII string

            Data:  "GROUP"

        Attribute: TITLE scalar

            Type:      46-byte null-terminated ASCII string

            Data:  "NHL player statistics for the 2013/2014 season"

        Attribute: VERSION scalar

            Type:      3-byte null-terminated ASCII string

            Data:  "1.0"

        Location:  1:1032

        Links:     1

    /season_2013_2014/top30  Dataset {30/Inf}

        Attribute: CLASS scalar

            Type:      5-byte null-terminated ASCII string

            Data:  "TABLE"

        Attribute: FIELD_0_FILL scalar

            Type:      native unsigned short

            Data:  0

        Attribute: FIELD_0_NAME scalar

            Type:      7-byte null-terminated ASCII string

            Data:  "assists"

        Attribute: FIELD_1_FILL scalar

            Type:      native unsigned char

            Data:  0

        Attribute: FIELD_1_NAME scalar

            Type:      12-byte null-terminated ASCII string

            Data:  "games_played"

        Attribute: FIELD_2_FILL scalar

            Type:      native unsigned short

            Data:  0

        Attribute: FIELD_2_NAME scalar

            Type:      5-byte null-terminated ASCII string

            Data:  "goals"

        Attribute: FIELD_3_FILL scalar

            Type:      1-byte null-terminated ASCII string

            Data:  ""

        Attribute: FIELD_3_NAME scalar

            Type:      6-byte null-terminated ASCII string

            Data:  "player"

        Attribute: FIELD_4_FILL scalar

            Type:      native unsigned short

            Data:  0

        Attribute: FIELD_4_NAME scalar

            Type:      6-byte null-terminated ASCII string

            Data:  "points"

        Attribute: FIELD_5_FILL scalar

            Type:      1-byte null-terminated ASCII string

            Data:  "C"

        Attribute: FIELD_5_NAME scalar

            Type:      8-byte null-terminated ASCII string

            Data:  "position"

        Attribute: FIELD_6_FILL scalar

            Type:      native double

            Data:  0

        Attribute: FIELD_6_NAME scalar

            Type:      22-byte null-terminated ASCII string

            Data:  "shifts_per_game_played"

        Attribute: FIELD_7_FILL scalar

            Type:      native double

            Data:  0

        Attribute: FIELD_7_NAME scalar

            Type:      19-byte null-terminated ASCII string

            Data:  "shooting_percentage"

        Attribute: NROWS scalar

            Type:      native long

            Data:  30

        Attribute: TITLE scalar

            Type:      20-byte null-terminated ASCII string

            Data:  "Top 30 point leaders"

        Attribute: VERSION scalar

            Type:      3-byte null-terminated ASCII string

            Data:  "2.7"

        Location:  1:2272

        Links:     1

        Chunks:    {1489} 65516 bytes

        Storage:   1320 logical bytes, 65516 allocated bytes, 2.01% utilization

        Type:      struct {

                       "assists"          +0    native unsigned short

                       "games_played"     +2    native unsigned char

                       "goals"            +3    native unsigned short

                       "player"           +5    20-byte null-terminated ASCII string

                       "points"           +25   native unsigned short

                       "position"         +27   1-byte null-terminated ASCII string

                       "shifts_per_game_played" +28   native double

                       "shooting_percentage" +36   native double

                   } 44 bytes



## Pandas hdfstore


```python
import pandas as pd
```


```python
store = pd.HDFStore('store.h5')
```


```python
df = pd.DataFrame(np.random.rand(5,5))
```


```python
store["df1"] = df
```


```python
df = pd.read_csv("playerstats-2013-2014-top30.csv", skiprows=1)
```


```python
store["df2"] = df
```


```python
store.keys()
```




    ['/df1', '/df2']




```python
'df2' in store
```




    True




```python
df = store["df1"]
```


```python
store.root
```




    / (RootGroup) ''
      children := ['df1' (Group), 'df2' (Group)]




```python
store.close()
```


```python
f = h5py.File("store.h5")
```


```python
f.visititems(lambda x, y: print(x, "\t" * int(3 - len(str(x))//8), y))
```

    df1 			 <HDF5 group "/df1" (4 members)>
    df1/axis0 		 <HDF5 dataset "axis0": shape (5,), type "<i8">
    df1/axis1 		 <HDF5 dataset "axis1": shape (5,), type "<i8">
    df1/block0_items 	 <HDF5 dataset "block0_items": shape (5,), type "<i8">
    df1/block0_values 	 <HDF5 dataset "block0_values": shape (5, 5), type "<f8">
    df2 			 <HDF5 group "/df2" (8 members)>
    df2/axis0 		 <HDF5 dataset "axis0": shape (21,), type "|S8">
    df2/axis1 		 <HDF5 dataset "axis1": shape (30,), type "<i8">
    df2/block0_items 	 <HDF5 dataset "block0_items": shape (3,), type "|S8">
    df2/block0_values 	 <HDF5 dataset "block0_values": shape (30, 3), type "<f8">
    df2/block1_items 	 <HDF5 dataset "block1_items": shape (14,), type "|S4">
    df2/block1_values 	 <HDF5 dataset "block1_values": shape (30, 14), type "<i8">
    df2/block2_items 	 <HDF5 dataset "block2_items": shape (4,), type "|S6">
    df2/block2_values 	 <HDF5 dataset "block2_values": shape (1,), type "|O8">



```python
f["/df2/block0_items"].value
```




    array(['S%', 'Shift/GP', 'FO%'],
          dtype='|S8')




```python
f["/df2/block0_values"][:3]
```




    array([[ 13.9,  24. ,  52.5],
           [ 15.2,  25.2,  49. ],
           [ 12.6,  25.1,  52.9]])




```python
f["/df2/block1_items"].value
```




    array(['Rank', 'GP', 'G', 'A', 'P', '+/-', 'PIM', 'PPG', 'PPP', 'SHG',
           'SHP', 'GW', 'OT', 'S'],
          dtype='|S4')




```python
f["/df2/block1_values"][:3, :5]
```




    array([[  1,  80,  36,  68, 104],
           [  2,  77,  31,  56,  87],
           [  3,  82,  28,  58,  86]])



# JSON


```python
data = ["string", 1.0, 2, None]
```


```python
data_json = json.dumps(data)
```


```python
data_json
```




    '["string", 1.0, 2, null]'




```python
data2 = json.loads(data_json)
```


```python
data
```




    ['string', 1.0, 2, None]




```python
data[0]
```




    'string'




```python
data = {"one": 1, "two": 2.0, "three": "three"}
```


```python
data_json = json.dumps(data)
```


```python
print(data_json)
```

    {"three": "three", "two": 2.0, "one": 1}



```python
data = json.loads(data_json)
```


```python
data["two"]
```




    2.0




```python
data["three"]
```




    u'three'




```python
data = {"one": [1],
        "two": [1, 2],
        "three": [1, 2, 3]}
```


```python
data_json = json.dumps(data, indent=True)
```


```python
print(data_json)
```

    {
     "three": [
      1,
      2,
      3
     ],
     "two": [
      1,
      2
     ],
     "one": [
      1
     ]
    }



```python
data = {"one": [1],
        "two": {"one": 1, "two": 2},
        "three": [(1,), (1, 2), (1, 2, 3)],
        "four": "a text string"}
```


```python
with open("data.json", "w") as f:
    json.dump(data, f)
```


```python
!cat data.json
```

    {"four": "a text string", "three": [[1], [1, 2], [1, 2, 3]], "two": {"two": 2, "one": 1}, "one": [1]}


```python
with open("data.json", "r") as f:
    data_from_file = json.load(f)
```


```python
data_from_file["two"]
```




    {u'one': 1, u'two': 2}




```python
data_from_file["three"]
```




    [[1], [1, 2], [1, 2, 3]]




```python
!head -n 20 tokyo-metro.json
```

    {

        "C": {

            "color": "#149848",

            "transfers": [

                [

                    "C3",

                    "F15"

                ],

                [

                    "C4",

                    "Z2"

                ],

                [

                    "C4",

                    "G2"

                ],

                [

                    "C7",

                    "M14"

                ],




```python
!wc tokyo-metro.json
```

        1471    1508   27638 tokyo-metro.json




```python
with open("tokyo-metro.json", "r") as f:
    data = json.load(f)
```


```python
data.keys()
```




    [u'C', u'G', u'F', u'H', u'M', u'N', u'T', u'Y', u'Z']




```python
data["C"].keys()
```




    [u'color', u'transfers', u'travel_times']




```python
data["C"]["color"]
```




    u'#149848'




```python
data["C"]["transfers"]
```




    [[u'C3', u'F15'],
     [u'C4', u'Z2'],
     [u'C4', u'G2'],
     [u'C7', u'M14'],
     [u'C7', u'N6'],
     [u'C7', u'G6'],
     [u'C8', u'M15'],
     [u'C8', u'H6'],
     [u'C9', u'H7'],
     [u'C9', u'Y18'],
     [u'C11', u'T9'],
     [u'C11', u'M18'],
     [u'C11', u'Z8'],
     [u'C12', u'M19'],
     [u'C18', u'H21']]




```python
[(s, e, tt) for s, e, tt in data["C"]["travel_times"] if tt == 1]
```




    [(u'C3', u'C4', 1), (u'C7', u'C8', 1), (u'C9', u'C10', 1)]




```python
data
```




    {u'C': {u'color': u'#149848',
      u'transfers': [[u'C3', u'F15'],
       [u'C4', u'Z2'],
       [u'C4', u'G2'],
       [u'C7', u'M14'],
       [u'C7', u'N6'],
       [u'C7', u'G6'],
       [u'C8', u'M15'],
       [u'C8', u'H6'],
       [u'C9', u'H7'],
       [u'C9', u'Y18'],
       [u'C11', u'T9'],
       [u'C11', u'M18'],
       [u'C11', u'Z8'],
       [u'C12', u'M19'],
       [u'C18', u'H21']],
      u'travel_times': [[u'C1', u'C2', 2],
       [u'C2', u'C3', 2],
       [u'C3', u'C4', 1],
       [u'C4', u'C5', 2],
       [u'C5', u'C6', 2],
       [u'C6', u'C7', 2],
       [u'C7', u'C8', 1],
       [u'C8', u'C9', 3],
       [u'C9', u'C10', 1],
       [u'C10', u'C11', 2],
       [u'C11', u'C12', 2],
       [u'C12', u'C13', 2],
       [u'C13', u'C14', 2],
       [u'C14', u'C15', 2],
       [u'C15', u'C16', 2],
       [u'C16', u'C17', 3],
       [u'C17', u'C18', 3],
       [u'C18', u'C19', 3]]},
     u'F': {u'color': u'#b96528',
      u'transfers': [[u'F1', u'Y1'],
       [u'F2', u'Y2'],
       [u'F3', u'Y3'],
       [u'F4', u'Y4'],
       [u'F5', u'Y5'],
       [u'F6', u'Y6'],
       [u'F7', u'Y7'],
       [u'F8', u'Y8'],
       [u'F9', u'Y9'],
       [u'F9', u'M25'],
       [u'F13', u'M9'],
       [u'F15', u'C3'],
       [u'F16', u'Z1'],
       [u'F16', u'G1']],
      u'travel_times': [[u'F1', u'F2', 3],
       [u'F2', u'F3', 2],
       [u'F3', u'F4', 3],
       [u'F4', u'F5', 2],
       [u'F5', u'F6', 2],
       [u'F6', u'F7', 2],
       [u'F7', u'F8', 2],
       [u'F8', u'F9', 2],
       [u'F9', u'F10', 3],
       [u'F10', u'F11', 2],
       [u'F11', u'F12', 2],
       [u'F12', u'F13', 2],
       [u'F13', u'F14', 3],
       [u'F14', u'F15', 2],
       [u'F15', u'F16', 2]]},
     u'G': {u'color': u'#f59230',
      u'transfers': [[u'G1', u'Z1'],
       [u'G1', u'F16'],
       [u'G2', u'Z2'],
       [u'G2', u'C4'],
       [u'G4', u'Z3'],
       [u'G5', u'M13'],
       [u'G5', u'Y16'],
       [u'G5', u'Z4'],
       [u'G5', u'N7'],
       [u'G6', u'N6'],
       [u'G6', u'M14'],
       [u'G6', u'C7'],
       [u'G9', u'M16'],
       [u'G9', u'H8'],
       [u'G11', u'T10'],
       [u'G12', u'Z9'],
       [u'G15', u'H16'],
       [u'G16', u'H17']],
      u'travel_times': [[u'G1', u'G2', 2],
       [u'G2', u'G3', 1],
       [u'G3', u'G4', 2],
       [u'G4', u'G5', 2],
       [u'G5', u'G6', 2],
       [u'G6', u'G7', 2],
       [u'G7', u'G8', 2],
       [u'G8', u'G9', 2],
       [u'G9', u'G10', 1],
       [u'G10', u'G11', 2],
       [u'G11', u'G12', 2],
       [u'G12', u'G13', 1],
       [u'G13', u'G14', 2],
       [u'G14', u'G15', 2],
       [u'G15', u'G16', 1],
       [u'G16', u'G17', 2],
       [u'G17', u'G18', 1],
       [u'G18', u'G19', 2]]},
     u'H': {u'color': u'#9cacb5',
      u'transfers': [[u'H6', u'M15'],
       [u'H6', u'C8'],
       [u'H7', u'Y18'],
       [u'H7', u'C9'],
       [u'H8', u'M16'],
       [u'H8', u'G9'],
       [u'H12', u'T11'],
       [u'H16', u'G15'],
       [u'H17', u'G16'],
       [u'H21', u'C18']],
      u'travel_times': [[u'H1', u'H2', 3],
       [u'H2', u'H3', 3],
       [u'H3', u'H4', 3],
       [u'H4', u'H5', 3],
       [u'H5', u'H6', 2],
       [u'H6', u'H7', 3],
       [u'H7', u'H8', 1],
       [u'H8', u'H9', 2],
       [u'H9', u'H10', 2],
       [u'H10', u'H11', 2],
       [u'H11', u'H12', 1],
       [u'H12', u'H13', 3],
       [u'H13', u'H14', 1],
       [u'H14', u'H15', 2],
       [u'H15', u'H16', 2],
       [u'H16', u'H17', 1],
       [u'H17', u'H18', 2],
       [u'H18', u'H19', 2],
       [u'H19', u'H20', 2],
       [u'H20', u'H21', 3]]},
     u'M': {u'color': u'#ff0000',
      u'transfers': [[u'M9', u'F13'],
       [u'M12', u'N8'],
       [u'M13', u'G5'],
       [u'M13', u'Y16'],
       [u'M13', u'Z4'],
       [u'M13', u'N7'],
       [u'M14', u'C7'],
       [u'M14', u'G6'],
       [u'M14', u'N6'],
       [u'M15', u'H6'],
       [u'M15', u'C8'],
       [u'M16', u'G9'],
       [u'M16', u'H8'],
       [u'M18', u'T9'],
       [u'M18', u'C11'],
       [u'M18', u'Z8'],
       [u'M19', u'C12'],
       [u'M22', u'N11'],
       [u'M25', u'Y9'],
       [u'M25', u'F9']],
      u'travel_times': [[u'M1', u'M2', 2],
       [u'M2', u'M3', 2],
       [u'M3', u'M4', 2],
       [u'M4', u'M5', 2],
       [u'M5', u'M6', 2],
       [u'M6', u'M7', 2],
       [u'M7', u'M8', 2],
       [u'M8', u'M9', 2],
       [u'M9', u'M10', 1],
       [u'M10', u'M11', 2],
       [u'M11', u'M12', 2],
       [u'M12', u'M13', 3],
       [u'M13', u'M14', 2],
       [u'M14', u'M15', 1],
       [u'M15', u'M16', 3],
       [u'M16', u'M17', 2],
       [u'M17', u'M18', 2],
       [u'M18', u'M19', 2],
       [u'M19', u'M20', 1],
       [u'M20', u'M21', 2],
       [u'M21', u'M22', 2],
       [u'M22', u'M23', 3],
       [u'M23', u'M24', 2],
       [u'M24', u'M25', 3],
       [u'm3', u'm4', 2],
       [u'm4', u'm5', 2],
       [u'm5', u'M6', 2]]},
     u'N': {u'color': u'#1aaca9',
      u'transfers': [[u'N1', u'T1'],
       [u'N2', u'T2'],
       [u'N3', u'T3'],
       [u'N6', u'G6'],
       [u'N6', u'M14'],
       [u'N6', u'C7'],
       [u'N7', u'Y16'],
       [u'N7', u'Z4'],
       [u'N7', u'G5'],
       [u'N7', u'M13'],
       [u'N8', u'M12'],
       [u'N9', u'Y14'],
       [u'N10', u'Y13'],
       [u'N10', u'T6'],
       [u'N11', u'M22']],
      u'travel_times': [[u'N1', u'N2', 2],
       [u'N2', u'N3', 2],
       [u'N3', u'N4', 2],
       [u'N4', u'N5', 2],
       [u'N5', u'N6', 2],
       [u'N6', u'N7', 2],
       [u'N7', u'N8', 2],
       [u'N8', u'N9', 2],
       [u'N9', u'N10', 2],
       [u'N10', u'N11', 2],
       [u'N11', u'N12', 3],
       [u'N12', u'N13', 2],
       [u'N13', u'N14', 2],
       [u'N14', u'N15', 3],
       [u'N15', u'N16', 1],
       [u'N16', u'N17', 3],
       [u'N17', u'N18', 2],
       [u'N18', u'N19', 2]]},
     u'T': {u'color': u'#1aa7d8',
      u'transfers': [[u'T6', u'N10'],
       [u'T6', u'Y13'],
       [u'T7', u'Z6'],
       [u'T9', u'M18'],
       [u'T9', u'C11'],
       [u'T9', u'Z8'],
       [u'T10', u'G11'],
       [u'T11', u'H12']],
      u'travel_times': [[u'T1', u'T2', 0],
       [u'T2', u'T3', 3],
       [u'T3', u'T4', 6],
       [u'T4', u'T5', 9],
       [u'T5', u'T6', 11],
       [u'T6', u'T7', 13],
       [u'T7', u'T8', 14],
       [u'T8', u'T9', 16],
       [u'T9', u'T10', 18],
       [u'T10', u'T11', 20],
       [u'T11', u'T12', 21],
       [u'T12', u'T13', 24],
       [u'T13', u'T14', 26],
       [u'T14', u'T15', 27],
       [u'T15', u'T16', 30],
       [u'T16', u'T17', 33],
       [u'T17', u'T18', 35],
       [u'T18', u'T19', 37],
       [u'T19', u'T20', 39],
       [u'T20', u'T21', 41],
       [u'T21', u'T22', 43],
       [u'T22', u'T23', 46],
       [u'T23', u'T24', 49]]},
     u'Y': {u'color': u'#ede7c3',
      u'transfers': [[u'Y1', u'F1'],
       [u'Y2', u'F2'],
       [u'Y3', u'F3'],
       [u'Y4', u'F4'],
       [u'Y5', u'F5'],
       [u'Y6', u'F6'],
       [u'Y7', u'F7'],
       [u'Y8', u'F8'],
       [u'Y9', u'F9'],
       [u'Y9', u'M25'],
       [u'Y13', u'T6'],
       [u'Y13', u'N10'],
       [u'Y14', u'N9'],
       [u'Y16', u'Z4'],
       [u'Y16', u'N7'],
       [u'Y16', u'G5'],
       [u'Y16', u'M13'],
       [u'Y18', u'H7'],
       [u'Y18', u'C9']],
      u'travel_times': [[u'Y1', u'Y2', 4],
       [u'Y2', u'Y3', 2],
       [u'Y3', u'Y4', 3],
       [u'Y4', u'Y5', 2],
       [u'Y5', u'Y6', 2],
       [u'Y6', u'Y7', 2],
       [u'Y7', u'Y8', 2],
       [u'Y8', u'Y9', 3],
       [u'Y9', u'Y10', 2],
       [u'Y10', u'Y11', 2],
       [u'Y11', u'Y12', 2],
       [u'Y12', u'Y13', 3],
       [u'Y13', u'Y14', 2],
       [u'Y14', u'Y15', 2],
       [u'Y15', u'Y16', 1],
       [u'Y16', u'Y17', 2],
       [u'Y17', u'Y18', 2],
       [u'Y18', u'Y19', 2],
       [u'Y19', u'Y20', 2],
       [u'Y20', u'Y21', 2],
       [u'Y21', u'Y22', 2],
       [u'Y22', u'Y23', 3],
       [u'Y23', u'Y24', 2]]},
     u'Z': {u'color': u'#a384bf',
      u'transfers': [[u'Z1', u'F16'],
       [u'Z1', u'G1'],
       [u'Z2', u'C4'],
       [u'Z2', u'G2'],
       [u'Z3', u'G4'],
       [u'Z4', u'Y16'],
       [u'Z4', u'N7'],
       [u'Z4', u'M13'],
       [u'Z4', u'G5'],
       [u'Z6', u'T7'],
       [u'Z8', u'M18'],
       [u'Z8', u'C11'],
       [u'Z8', u'T9'],
       [u'Z9', u'G12']],
      u'travel_times': [[u'Z1', u'Z2', 3],
       [u'Z2', u'Z3', 2],
       [u'Z3', u'Z4', 2],
       [u'Z4', u'Z5', 2],
       [u'Z5', u'Z6', 2],
       [u'Z6', u'Z7', 2],
       [u'Z7', u'Z8', 2],
       [u'Z8', u'Z9', 2],
       [u'Z9', u'Z10', 3],
       [u'Z10', u'Z11', 3],
       [u'Z11', u'Z12', 3],
       [u'Z12', u'Z13', 2],
       [u'Z13', u'Z14', 2]]}}




```python
!ls -lh tokyo-metro.json
```

    -rw-r--r--@ 1 rob  staff    27K Jul 20 00:01 tokyo-metro.json




```python
data_pack = msgpack.packb(data)
```


```python
del data
```


```python
type(data_pack)
```




    str




```python
len(data_pack)
```




    3021




```python
with open("tokyo-metro.msgpack", "wb") as f:
    f.write(data_pack)
```


```python
!ls -lh tokyo-metro.msgpack
```

    -rw-r--r--@ 1 rob  staff   3.0K Aug  3 23:46 tokyo-metro.msgpack




```python
with open("tokyo-metro.msgpack", "rb") as f:
    data_msgpack = f.read()
    data = msgpack.unpackb(data_msgpack)
```


```python
list(data.keys())
```




    ['C', 'G', 'F', 'H', 'M', 'N', 'T', 'Y', 'Z']




```python
with open("tokyo-metro.pickle", "wb") as f:
    cPickle.dump(data, f)
```


```python
del data
```


```python
!ls -lh tokyo-metro.pickle
```

    -rw-r--r--@ 1 rob  staff    11K Aug  3 23:46 tokyo-metro.pickle




```python
with open("tokyo-metro.pickle", "rb") as f:
    data = pickle.load(f)
```


```python
data.keys()
```




    ['C', 'G', 'F', 'H', 'M', 'N', 'T', 'Y', 'Z']



# Versions


```python
%reload_ext version_information
```


```python
%version_information numpy, pandas, csv, json, tables, h5py, msgpack
```




<table><tr><th>Software</th><th>Version</th></tr><tr><td>Python</td><td>2.7.10 64bit [GCC 4.2.1 (Apple Inc. build 5577)]</td></tr><tr><td>IPython</td><td>3.2.1</td></tr><tr><td>OS</td><td>Darwin 14.1.0 x86_64 i386 64bit</td></tr><tr><td>numpy</td><td>1.9.2</td></tr><tr><td>pandas</td><td>0.16.2</td></tr><tr><td>csv</td><td>1.0</td></tr><tr><td>json</td><td>2.0.9</td></tr><tr><td>tables</td><td>3.2.0</td></tr><tr><td>h5py</td><td>2.5.0</td></tr><tr><td>msgpack</td><td>The 'msgpack' distribution was not found and is required by the application</td></tr></table>
