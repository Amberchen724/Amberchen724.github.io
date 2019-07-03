

```python
import os
import glob
import pandas as pd 
import geopandas as gpd
from geopandas import GeoDataFrame
import os
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pylab as plt
import math
import seaborn as sns

from urllib.error import HTTPError
import urllib.request as request
try:
    from urllib import urlretrieve as urlretrieve
except ImportError:
    from urllib.request import urlretrieve as urlretrieve

from statsmodels.graphics.api import qqplot
import pylab as pl
 
pd.options.mode.chained_assignment = None
%matplotlib inline
```

### Table of Contents:
* [Merge Census Data with Geospatial Data](#1)
* [EDA](#2)
* [Clustering](#3)
    * [2016 Data](#4)
        * [K-means](#4)
        * [GMM](#5)
    * [2000 Data](#6)
        * [K-means](#6)
        * [GMM](#7)
    * [1990 Data](#8)
        * [K-means](#8)
        * [GMM](#9)
* [Clusters Difference](#10)
* [Identify the changes of each variables from 2000 to 2016](#11)
* [Summary and Analysis](#12)

<a id='1'></a>
## 1.Merge Census Data with Geospatial Data


```python
masterdata = os.getenv("Master")
if masterdata is None:
    os.environ["Master"] = "{}/ML4C/Project".format(os.getenv("HOME"))
    masterdata = os.getenv("Master")
    print("Warning: Master environmental variable not found and set by code, please review!")
print("Master: {}".format(masterdata))
```

    Master: /Users/amber/ML4C/Project



```python
def getGeoDataFrameFromShpFileZipUrl(url):
    '''
    This function downloads the zip file, unzips it into the dorectory 
    pointed to by PUIdata environment variable. Then it 
    reads it into a gepandas dataframe
    '''
    
    folderName = 'shape'+ \
        str(len(os.listdir(os.getenv('Master')))+1)
    os.makedirs(os.getenv('Master') + '/' + folderName)
    urlretrieve(url, "region.zip")
    os.system('unzip -d $Master'+'/'+folderName+' region.zip')
    filenames = [f for f in os.listdir(os.getenv('Master') + '/' + folderName) if f.endswith('.shp') ]
    shapeFile = filenames[0]
    shapeFilePath = os.getenv('Master') + '/' + folderName + '/' + shapeFile
    return gpd.GeoDataFrame.from_file(shapeFilePath)
```


```python
NYCzip=gpd.read_file('Data/14000.shp')
```


```python
NYCzip.rename(columns={"GEO_ID": "GEOID"},inplace=True)
NYCzip.GEOID = NYCzip.GEOID.astype(str).str[-11:]
NYCzip.GEOID = NYCzip.GEOID.astype(int)
```


```python
cols = ['GEOID']
NYCzip = NYCzip.loc[:,cols]
#NYCzipgdp.plot(column='GEOID',legend = True)
NYCzip.shape
```




    (2166, 1)




```python
NYzip_url = 'https://planninglabs.carto.com/api/v2/sql?filename=region&q=SELECT%20%2A%20FROM%20region_censustract_v0&format=SHP'
NYzip = getGeoDataFrameFromShpFileZipUrl(NYzip_url)
```


```python
NYzip.rename(columns={"geoid": "GEOID"},inplace=True)
NYzip.GEOID = NYzip.GEOID.astype(int)
cols = ['GEOID','geometry']
NYCzipgdp = NYzip.loc[:,cols]

```


```python
NYCzipgdp = NYCzipgdp.merge(NYCzip,on='GEOID')
NYCzipgdp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36081113900</td>
      <td>POLYGON ((-73.79190199967752 40.76893599959674...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36061025700</td>
      <td>POLYGON ((-73.95068000038171 40.81084300040413...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36047000502</td>
      <td>POLYGON ((-73.99136499969497 40.69701000031133...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36081015802</td>
      <td>POLYGON ((-73.81497399975029 40.68669500037402...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36047116000</td>
      <td>POLYGON ((-73.88618300014529 40.66752100019399...</td>
    </tr>
  </tbody>
</table>
</div>




```python
Censuspd = pd.read_stata('Data/UDP_NYC_Variables.dta')
Censuspdgdp = gpd.GeoDataFrame(Censuspd)
```


```python
for column in Censuspdgdp.columns:
    Censuspdgdp[column] = pd.to_numeric(Censuspdgdp[column].values, errors='coerce')
```


```python
Censuspdgdp.rename(columns={'GEOid2': "GEOID"},inplace=True)
Censuspdgdp.tail(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID</th>
      <th>hh16</th>
      <th>moe_hh16</th>
      <th>per_col00</th>
      <th>per_nonwhite00</th>
      <th>per_rent00</th>
      <th>vli2000</th>
      <th>li2000</th>
      <th>mi2000</th>
      <th>hmi2000</th>
      <th>...</th>
      <th>pop90</th>
      <th>per_col90</th>
      <th>moe_hu16</th>
      <th>hinc16</th>
      <th>moe_hinc16</th>
      <th>popgrowth</th>
      <th>hinc00</th>
      <th>hinc90</th>
      <th>pct_ch_hinc90_00</th>
      <th>pct_ch_hinc00_16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5336</th>
      <td>36119984000</td>
      <td>92.0</td>
      <td>37.0</td>
      <td>0.121495</td>
      <td>0.517544</td>
      <td>0.0</td>
      <td>0.220431</td>
      <td>0.394953</td>
      <td>-1.490116e-08</td>
      <td>0.019975</td>
      <td>...</td>
      <td>749.0</td>
      <td>0.255689</td>
      <td>48.0</td>
      <td>38929.0</td>
      <td>16474.0</td>
      <td>-54.0</td>
      <td>46444.95</td>
      <td>55472.790653</td>
      <td>-0.162744</td>
      <td>-0.161825</td>
    </tr>
    <tr>
      <th>5337</th>
      <td>36119985000</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 89 columns</p>
</div>




```python
data_census = NYCzipgdp.merge(Censuspdgdp,on='GEOID')
data_census.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID</th>
      <th>geometry</th>
      <th>hh16</th>
      <th>moe_hh16</th>
      <th>per_col00</th>
      <th>per_nonwhite00</th>
      <th>per_rent00</th>
      <th>vli2000</th>
      <th>li2000</th>
      <th>mi2000</th>
      <th>...</th>
      <th>pop90</th>
      <th>per_col90</th>
      <th>moe_hu16</th>
      <th>hinc16</th>
      <th>moe_hinc16</th>
      <th>popgrowth</th>
      <th>hinc00</th>
      <th>hinc90</th>
      <th>pct_ch_hinc90_00</th>
      <th>pct_ch_hinc00_16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2159</th>
      <td>36061021703</td>
      <td>POLYGON ((-73.94607800039937 40.82126399983373...</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>0.134557</td>
      <td>0.989928</td>
      <td>1.000000</td>
      <td>0.481906</td>
      <td>0.208198</td>
      <td>0.064489</td>
      <td>...</td>
      <td>0.854182</td>
      <td>0.000095</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.006433</td>
      <td>34971.10032</td>
      <td>26333.700869</td>
      <td>0.327998</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2160</th>
      <td>36061021800</td>
      <td>POLYGON ((-73.94872800029694 40.80333100022552...</td>
      <td>2876.0</td>
      <td>150.0</td>
      <td>0.152688</td>
      <td>0.983061</td>
      <td>0.942688</td>
      <td>0.513238</td>
      <td>0.179940</td>
      <td>0.076447</td>
      <td>...</td>
      <td>4258.000000</td>
      <td>0.075093</td>
      <td>74.0</td>
      <td>48429.0</td>
      <td>6061.0</td>
      <td>2092.000000</td>
      <td>32256.70000</td>
      <td>16680.450280</td>
      <td>0.933803</td>
      <td>0.501363</td>
    </tr>
    <tr>
      <th>2161</th>
      <td>36061021900</td>
      <td>POLYGON ((-73.9554510000636 40.81937700004859,...</td>
      <td>2083.0</td>
      <td>115.0</td>
      <td>0.057858</td>
      <td>0.976335</td>
      <td>0.988011</td>
      <td>0.621887</td>
      <td>0.177527</td>
      <td>0.049810</td>
      <td>...</td>
      <td>5085.000000</td>
      <td>0.055690</td>
      <td>81.0</td>
      <td>21585.0</td>
      <td>5374.0</td>
      <td>-8.000000</td>
      <td>25044.40000</td>
      <td>25103.939512</td>
      <td>-0.002372</td>
      <td>-0.138131</td>
    </tr>
    <tr>
      <th>2162</th>
      <td>36061022102</td>
      <td>POLYGON ((-73.94515500010318 40.82087599992749...</td>
      <td>1117.0</td>
      <td>97.0</td>
      <td>0.282637</td>
      <td>0.974572</td>
      <td>0.827789</td>
      <td>0.439632</td>
      <td>0.187200</td>
      <td>0.079572</td>
      <td>...</td>
      <td>1408.000000</td>
      <td>0.161512</td>
      <td>69.0</td>
      <td>41635.0</td>
      <td>13171.0</td>
      <td>288.000000</td>
      <td>43768.25000</td>
      <td>35593.500000</td>
      <td>0.229670</td>
      <td>-0.048740</td>
    </tr>
    <tr>
      <th>2163</th>
      <td>36047152200</td>
      <td>POLYGON ((-73.96408699991511 40.6444069999831,...</td>
      <td>1064.0</td>
      <td>59.0</td>
      <td>0.422212</td>
      <td>0.660112</td>
      <td>0.521779</td>
      <td>0.129393</td>
      <td>0.100985</td>
      <td>0.064595</td>
      <td>...</td>
      <td>3171.000000</td>
      <td>1.036555</td>
      <td>49.0</td>
      <td>85147.0</td>
      <td>22920.0</td>
      <td>89.000000</td>
      <td>84499.12500</td>
      <td>98343.495042</td>
      <td>-0.140776</td>
      <td>0.007667</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 90 columns</p>
</div>



<a id='2'></a>
## 2. EDA


```python
data_census.shape
```




    (2164, 90)




```python
# Using regressor of 2016
cols_16 = []
for column in data_census.columns:
    if '16' in column:
        cols_16.append(column)
cols_16 
print(len(cols_16),cols_16 )
```

    34 ['hh16', 'moe_hh16', 'pop16', 'moe_pop16', 'ch_all_li_count00_16', 'moveinincd16', 'co_med_indinc16', 'per_limove16', 'mrent16', 'moe_mrent16', 'mhval16', 'moe_mhval16', 'mhval00_16', 'mrent00_16', 'hu16', 'ohu16', 'moe_owner16', 'rhu16', 'moe_renter16', 'per_rent16', 'per_nonwhite16', 'vli2016', 'li2016', 'mi2016', 'hmi2016', 'hi2016', 'vhi2016', 'per_all_li16', 'all_li_count16', 'per_col_16', 'moe_hu16', 'hinc16', 'moe_hinc16', 'pct_ch_hinc00_16']



```python
# Create a new dataframe with Geoid and geometry for variable from 2016
df_16 = data_census[['hh16', 'moe_hh16', 'pop16', 'moe_pop16', 'ch_all_li_count00_16', 'moveinincd16', 'co_med_indinc16', 'per_limove16', 'mrent16', 'moe_mrent16', 'mhval16', 'moe_mhval16', 'mhval00_16', 'mrent00_16', 'hu16', 'ohu16', 'moe_owner16', 'rhu16', 'moe_renter16', 'per_rent16', 'per_nonwhite16', 'vli2016', 'li2016', 'mi2016', 'hmi2016', 'hi2016', 'vhi2016', 'per_all_li16', 'all_li_count16', 'per_col_16', 'moe_hu16', 'hinc16', 'moe_hinc16', 'pct_ch_hinc00_16']]
df_16.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>hh16</th>
      <th>moe_hh16</th>
      <th>pop16</th>
      <th>moe_pop16</th>
      <th>ch_all_li_count00_16</th>
      <th>moveinincd16</th>
      <th>co_med_indinc16</th>
      <th>per_limove16</th>
      <th>mrent16</th>
      <th>moe_mrent16</th>
      <th>...</th>
      <th>hmi2016</th>
      <th>hi2016</th>
      <th>vhi2016</th>
      <th>per_all_li16</th>
      <th>all_li_count16</th>
      <th>per_col_16</th>
      <th>moe_hu16</th>
      <th>hinc16</th>
      <th>moe_hinc16</th>
      <th>pct_ch_hinc00_16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1369.0</td>
      <td>76.0</td>
      <td>3238.0</td>
      <td>291.0</td>
      <td>133.823456</td>
      <td>245.700</td>
      <td>59758.0</td>
      <td>0.395823</td>
      <td>1546.0</td>
      <td>115.0</td>
      <td>...</td>
      <td>0.121232</td>
      <td>0.116617</td>
      <td>0.275688</td>
      <td>0.421141</td>
      <td>576.541809</td>
      <td>0.332690</td>
      <td>47.0</td>
      <td>60655.0</td>
      <td>8461.0</td>
      <td>-0.217644</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1803.0</td>
      <td>102.0</td>
      <td>4511.0</td>
      <td>428.0</td>
      <td>-126.051941</td>
      <td>324.583</td>
      <td>75513.0</td>
      <td>0.555169</td>
      <td>1624.0</td>
      <td>236.0</td>
      <td>...</td>
      <td>0.072037</td>
      <td>0.103051</td>
      <td>0.352138</td>
      <td>0.408829</td>
      <td>737.118958</td>
      <td>0.641879</td>
      <td>47.0</td>
      <td>76832.0</td>
      <td>4932.0</td>
      <td>1.463107</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1528.0</td>
      <td>92.0</td>
      <td>2618.0</td>
      <td>232.0</td>
      <td>7.001373</td>
      <td>458.835</td>
      <td>50640.0</td>
      <td>0.125220</td>
      <td>1998.0</td>
      <td>171.0</td>
      <td>...</td>
      <td>0.041985</td>
      <td>0.057042</td>
      <td>0.650256</td>
      <td>0.203195</td>
      <td>310.481598</td>
      <td>0.866088</td>
      <td>31.0</td>
      <td>100805.0</td>
      <td>8522.0</td>
      <td>0.069335</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1235.0</td>
      <td>58.0</td>
      <td>4862.0</td>
      <td>405.0</td>
      <td>-100.156677</td>
      <td>337.804</td>
      <td>59758.0</td>
      <td>0.578571</td>
      <td>1467.0</td>
      <td>109.0</td>
      <td>...</td>
      <td>0.078507</td>
      <td>0.122173</td>
      <td>0.363464</td>
      <td>0.367679</td>
      <td>454.083557</td>
      <td>0.188823</td>
      <td>46.0</td>
      <td>69453.0</td>
      <td>15548.0</td>
      <td>0.131579</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896.0</td>
      <td>48.0</td>
      <td>2031.0</td>
      <td>203.0</td>
      <td>124.415955</td>
      <td>56.529</td>
      <td>50640.0</td>
      <td>0.757346</td>
      <td>780.0</td>
      <td>153.0</td>
      <td>...</td>
      <td>0.071625</td>
      <td>0.129686</td>
      <td>0.152520</td>
      <td>0.616670</td>
      <td>552.536011</td>
      <td>0.144699</td>
      <td>21.0</td>
      <td>28750.0</td>
      <td>7007.0</td>
      <td>-0.148336</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>



#### data exploration


```python
df_16= df_16.replace('nan',0)
df_16= df_16.replace(np.nan,0)
df_16= df_16.replace('NaN',0)
df_16.shape
```




    (2164, 34)




```python
# drop those variable that have more than 50 missing values 
#df_16_new = df_16.drop(['ch_all_li_count00_16','moe_mrent16','moe_mhval16','mhval00_16','mrent00_16','hinc16','moe_hinc16','pct_ch_hinc00_16'], axis=1)
#df_16_new.head()

```


```python
corr = df_16.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr)
plt.yticks(rotation=0, size=7)
plt.xticks(rotation=90, size=7)
plt.show()
```


![png](featureimportanceandclustering_files/featureimportanceandclustering_22_0.png)



```python
rel_vars = corr.mhval16 [(corr.mhval16>0.4)]
rel_cols = list(rel_vars.index.values)

corr2 = df_16[rel_cols].corr()
plt.figure(figsize=(8,8))
hm = sns.heatmap(corr2, annot=True, annot_kws={'size':10})
plt.yticks(rotation=0, size=10)
plt.xticks(rotation=90, size=10)
plt.title("correlation heatmap 2016")
plt.show()
```


![png](featureimportanceandclustering_files/featureimportanceandclustering_23_0.png)


#### After check the variable dictionary, the top three correlation variables are : per_col_16 --- "Percent college educated" /  mrent16 ---"Median house value" / vhi2016 --- "Share of very high income households". 


```python
# Create a new dataframe with Geoid and geometry for variable from 2016
df_16_geo = data_census[['GEOID', 'geometry','hh16', 'moe_hh16', 'pop16', 'moe_pop16', 'ch_all_li_count00_16', 'moveinincd16', 'co_med_indinc16', 'per_limove16', 'mrent16', 'moe_mrent16', 'mhval16', 'moe_mhval16', 'mhval00_16', 'mrent00_16', 'hu16', 'ohu16', 'moe_owner16', 'rhu16', 'moe_renter16', 'per_rent16', 'per_nonwhite16', 'vli2016', 'li2016', 'mi2016', 'hmi2016', 'hi2016', 'vhi2016', 'per_all_li16', 'all_li_count16', 'per_col_16', 'moe_hu16', 'hinc16', 'moe_hinc16', 'pct_ch_hinc00_16']]
```


```python
df_16_geo= df_16_geo.replace('nan',0)
df_16_geo= df_16_geo.replace(np.nan,0)
df_16_geo= df_16_geo.replace('NaN',0)
```

<a id='3'></a>
## 3. Clustering 

<a id='4'></a>
## 3.1. 2016 Data - K-means


```python
# With 2016 data

print(df_16_geo.shape)

cols_X = [i for i in df_16_geo.columns if (i != 'mhval16')&(i != 'GEOID')&(i !='geometry')]
print(cols_X)
```

    (2164, 36)
    ['hh16', 'moe_hh16', 'pop16', 'moe_pop16', 'ch_all_li_count00_16', 'moveinincd16', 'co_med_indinc16', 'per_limove16', 'mrent16', 'moe_mrent16', 'moe_mhval16', 'mhval00_16', 'mrent00_16', 'hu16', 'ohu16', 'moe_owner16', 'rhu16', 'moe_renter16', 'per_rent16', 'per_nonwhite16', 'vli2016', 'li2016', 'mi2016', 'hmi2016', 'hi2016', 'vhi2016', 'per_all_li16', 'all_li_count16', 'per_col_16', 'moe_hu16', 'hinc16', 'moe_hinc16', 'pct_ch_hinc00_16']



```python
X = np.asarray(df_16_geo[cols_X])
print(X.shape)
y = np.asarray(df_16_geo['mhval16'])
y
```

    (2164, 33)





    array([ 557700.,  827200.,  490200., ...,       0.,  537900., 1146400.])




```python
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def elbow(data,K):
#data is your input as numpy form
#K is a list of number of clusters you would like to show.
    # Run the KMeans model and save all the results for each number of clusters
    KM = [KMeans(n_clusters=k).fit(data) for k in K]
    
    # Save the centroids for each model with a increasing k
    centroids = [k.cluster_centers_ for k in KM]

    # For each k, get the distance between the data with each center. 
    D_k = [cdist(data, cent, 'euclidean') for cent in centroids]
    
    # But we only need the distance to the nearest centroid since we only calculate dist(x,ci) for its own cluster.
    globals()['dist'] = [np.min(D,axis=1) for D in D_k]
    
    # Calculate the Average SSE.
    avgWithinSS = [sum(d)/data.shape[0] for d in dist]
    
    
    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, avgWithinSS, 'b*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')
    plt.show()
    
    
    # Total with-in sum of square plot. Another way to show the result.
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(data)**2)/data.shape[0]
    bss = tss-wcss
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, bss/tss*100, 'b*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained')
    plt.title('Elbow for KMeans clustering')
    plt.show()
```


```python
elbow(X, range(2,11))
```


![png](featureimportanceandclustering_files/featureimportanceandclustering_32_0.png)



![png](featureimportanceandclustering_files/featureimportanceandclustering_32_1.png)



```python
range_n_clusters = range(2,20)

for n_clusters in range_n_clusters:
    km = KMeans(n_clusters = n_clusters)
    res=km.fit(X)
    silhouette_avg = silhouette_score(X, res.labels_)
    print("For n_clusters = {},".format(n_clusters)+" the average silhouette_score is : {}".format(silhouette_avg))
    
        
```

    For n_clusters = 2, the average silhouette_score is : 0.7029014192702603
    For n_clusters = 3, the average silhouette_score is : 0.5869307469791986
    For n_clusters = 4, the average silhouette_score is : 0.45975127475277344
    For n_clusters = 5, the average silhouette_score is : 0.4367914972440461
    For n_clusters = 6, the average silhouette_score is : 0.3278261069304003
    For n_clusters = 7, the average silhouette_score is : 0.3348276259908766
    For n_clusters = 8, the average silhouette_score is : 0.30853247571775466
    For n_clusters = 9, the average silhouette_score is : 0.33582293856776
    For n_clusters = 10, the average silhouette_score is : 0.32533108145814593
    For n_clusters = 11, the average silhouette_score is : 0.30950846730906695
    For n_clusters = 12, the average silhouette_score is : 0.3121402545930143
    For n_clusters = 13, the average silhouette_score is : 0.3259077640740861
    For n_clusters = 14, the average silhouette_score is : 0.32622576726186386
    For n_clusters = 15, the average silhouette_score is : 0.2893491558424447
    For n_clusters = 16, the average silhouette_score is : 0.27794260583915
    For n_clusters = 17, the average silhouette_score is : 0.29091863431954795
    For n_clusters = 18, the average silhouette_score is : 0.2890158974851763
    For n_clusters = 19, the average silhouette_score is : 0.27164928726337484



```python
n=4 # number of clusters
dd=X #data
tar=y # real target

km=KMeans(n_clusters=n)
res=km.fit(dd)
```


```python
y_clusters = res.labels_
y_clusters
```




    array([3, 3, 3, ..., 1, 1, 3], dtype=int32)




```python
df_16_geo['clusters'] =y_clusters
df_16_geo['clusters'].unique()
df_16_geo.to_csv('clusters_16.csv')
df_16_geo.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID</th>
      <th>geometry</th>
      <th>hh16</th>
      <th>moe_hh16</th>
      <th>pop16</th>
      <th>moe_pop16</th>
      <th>ch_all_li_count00_16</th>
      <th>moveinincd16</th>
      <th>co_med_indinc16</th>
      <th>per_limove16</th>
      <th>...</th>
      <th>hi2016</th>
      <th>vhi2016</th>
      <th>per_all_li16</th>
      <th>all_li_count16</th>
      <th>per_col_16</th>
      <th>moe_hu16</th>
      <th>hinc16</th>
      <th>moe_hinc16</th>
      <th>pct_ch_hinc00_16</th>
      <th>clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2161</th>
      <td>36061021900</td>
      <td>POLYGON ((-73.9554510000636 40.81937700004859,...</td>
      <td>2083.0</td>
      <td>115.0</td>
      <td>6415.0</td>
      <td>755.0</td>
      <td>-57.557861</td>
      <td>506.445</td>
      <td>75513.0</td>
      <td>0.719861</td>
      <td>...</td>
      <td>0.032178</td>
      <td>0.065179</td>
      <td>0.836641</td>
      <td>1742.723755</td>
      <td>0.186234</td>
      <td>81.0</td>
      <td>21585.0</td>
      <td>5374.0</td>
      <td>-0.138131</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2162</th>
      <td>36061022102</td>
      <td>POLYGON ((-73.94515500010318 40.82087599992749...</td>
      <td>1117.0</td>
      <td>97.0</td>
      <td>2451.0</td>
      <td>291.0</td>
      <td>22.305359</td>
      <td>245.920</td>
      <td>75513.0</td>
      <td>0.788569</td>
      <td>...</td>
      <td>0.066406</td>
      <td>0.204434</td>
      <td>0.593489</td>
      <td>662.927551</td>
      <td>0.519689</td>
      <td>69.0</td>
      <td>41635.0</td>
      <td>13171.0</td>
      <td>-0.048740</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2163</th>
      <td>36047152200</td>
      <td>POLYGON ((-73.96408699991511 40.6444069999831,...</td>
      <td>1064.0</td>
      <td>59.0</td>
      <td>3496.0</td>
      <td>370.0</td>
      <td>30.123215</td>
      <td>173.047</td>
      <td>50640.0</td>
      <td>0.607703</td>
      <td>...</td>
      <td>0.120580</td>
      <td>0.548916</td>
      <td>0.266917</td>
      <td>284.000000</td>
      <td>0.561866</td>
      <td>49.0</td>
      <td>85147.0</td>
      <td>22920.0</td>
      <td>0.007667</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 37 columns</p>
</div>




```python
df_16_geo.to_file(driver = 'ESRI Shapefile', filename= "2016clusterresult.shp")
```


```python
f, ax = plt.subplots(figsize=(10,10))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
df_16_geo.plot(column='clusters',cmap='viridis',alpha=1,
                  linewidth=0.1,ax=ax, categorical=True,legend=True, k=10)
plt.title("median property price clustering map for 2016 (k-means)")
```




    Text(0.5, 1.0, 'median property price clustering map for 2016 (k-means)')




![png](featureimportanceandclustering_files/featureimportanceandclustering_38_1.png)


#### Start from here, we are trying to find out how those important features changes over time and how they contributed to different clusters in different years.  


```python
df_16_geo_0 = df_16_geo.loc[df_16_geo['clusters'] == 0]
df_16_geo_1 = df_16_geo.loc[df_16_geo['clusters'] == 1]
df_16_geo_2 = df_16_geo.loc[df_16_geo['clusters'] == 2]
df_16_geo_3 = df_16_geo.loc[df_16_geo['clusters'] == 3]
```


```python
print ('mean value of median house price per each cluster:')
print (df_16_geo_0['mhval16'].mean())
print (df_16_geo_1['mhval16'].mean())
print (df_16_geo_2['mhval16'].mean())
print (df_16_geo_3['mhval16'].mean())
```

    mean value of median house price per each cluster:
    813797.7653631285
    420136.1762615494
    1233982.142857143
    640749.2727272727



```python
print ('mean value of percentage of college educated per each cluster:')
print (df_16_geo_0['per_col_16'].mean())
print (df_16_geo_1['per_col_16'].mean())
print (df_16_geo_2['per_col_16'].mean())
print (df_16_geo_3['per_col_16'].mean())
```

    mean value of percentage of college educated per each cluster:
    0.47267938265837106
    0.2860536255212481
    0.7029302641749382
    0.39031642314385284



```python
print ('mean value of share of very high income households per each cluster:')
print (df_16_geo_0['vhi2016'].mean())
print (df_16_geo_1['vhi2016'].mean())
print (df_16_geo_2['vhi2016'].mean())
print (df_16_geo_3['vhi2016'].mean())
```

    mean value of share of very high income households per each cluster:
    0.35707405326086716
    0.33706655689298726
    0.502770443047796
    0.3448041877963326



```python
print ('mean value of share of median income households per each cluster:')
print (df_16_geo_0['hinc16'].mean())
print (df_16_geo_1['hinc16'].mean())
print (df_16_geo_2['hinc16'].mean())
print (df_16_geo_3['hinc16'].mean())
```

    mean value of share of median income households per each cluster:
    68664.26815642459
    56895.0277185501
    108565.53571428571
    60879.696363636365



```python
print ('mean value of median rent per each cluster:')
print (df_16_geo_0['mrent16'].mean())
print (df_16_geo_1['mrent16'].mean())
print (df_16_geo_2['mrent16'].mean())
print (df_16_geo_3['mrent16'].mean())
```

    mean value of median rent per each cluster:
    1541.68156424581
    1262.4463397299219
    2093.5714285714284
    1474.1472727272728





```python

```

<a id='5'></a>
## 3.1. 2016 Data - GaussianMixture


```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=4).fit(X)
labels = gmm.predict(X)
```


```python
y_clusters = labels
y_clusters
```




    array([2, 1, 1, ..., 1, 2, 2])




```python
df_16_geo['gmmclusters'] =y_clusters
df_16_geo['gmmclusters'].unique()
df_16_geo.to_csv('clusters_16_gmm.csv')
df_16_geo.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID</th>
      <th>geometry</th>
      <th>hh16</th>
      <th>moe_hh16</th>
      <th>pop16</th>
      <th>moe_pop16</th>
      <th>ch_all_li_count00_16</th>
      <th>moveinincd16</th>
      <th>co_med_indinc16</th>
      <th>per_limove16</th>
      <th>...</th>
      <th>vhi2016</th>
      <th>per_all_li16</th>
      <th>all_li_count16</th>
      <th>per_col_16</th>
      <th>moe_hu16</th>
      <th>hinc16</th>
      <th>moe_hinc16</th>
      <th>pct_ch_hinc00_16</th>
      <th>clusters</th>
      <th>gmmclusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2161</th>
      <td>36061021900</td>
      <td>POLYGON ((-73.9554510000636 40.81937700004859,...</td>
      <td>2083.0</td>
      <td>115.0</td>
      <td>6415.0</td>
      <td>755.0</td>
      <td>-57.557861</td>
      <td>506.445</td>
      <td>75513.0</td>
      <td>0.719861</td>
      <td>...</td>
      <td>0.065179</td>
      <td>0.836641</td>
      <td>1742.723755</td>
      <td>0.186234</td>
      <td>81.0</td>
      <td>21585.0</td>
      <td>5374.0</td>
      <td>-0.138131</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2162</th>
      <td>36061022102</td>
      <td>POLYGON ((-73.94515500010318 40.82087599992749...</td>
      <td>1117.0</td>
      <td>97.0</td>
      <td>2451.0</td>
      <td>291.0</td>
      <td>22.305359</td>
      <td>245.920</td>
      <td>75513.0</td>
      <td>0.788569</td>
      <td>...</td>
      <td>0.204434</td>
      <td>0.593489</td>
      <td>662.927551</td>
      <td>0.519689</td>
      <td>69.0</td>
      <td>41635.0</td>
      <td>13171.0</td>
      <td>-0.048740</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2163</th>
      <td>36047152200</td>
      <td>POLYGON ((-73.96408699991511 40.6444069999831,...</td>
      <td>1064.0</td>
      <td>59.0</td>
      <td>3496.0</td>
      <td>370.0</td>
      <td>30.123215</td>
      <td>173.047</td>
      <td>50640.0</td>
      <td>0.607703</td>
      <td>...</td>
      <td>0.548916</td>
      <td>0.266917</td>
      <td>284.000000</td>
      <td>0.561866</td>
      <td>49.0</td>
      <td>85147.0</td>
      <td>22920.0</td>
      <td>0.007667</td>
      <td>3</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 38 columns</p>
</div>




```python
f, ax = plt.subplots(figsize=(10,10))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
df_16_geo.plot(column='gmmclusters',cmap='viridis',alpha=1,
                  linewidth=0.1,ax=ax, categorical=True,legend=True, k=10)
plt.title("median property price clustering map for 2016(GMM)")
```




    Text(0.5, 1.0, 'median property price clustering map for 2016(GMM)')




![png](featureimportanceandclustering_files/featureimportanceandclustering_52_1.png)



```python
df_16_geo_0_gmm = df_16_geo.loc[df_16_geo['gmmclusters'] == 0]
df_16_geo_1_gmm = df_16_geo.loc[df_16_geo['gmmclusters'] == 1]
df_16_geo_2_gmm = df_16_geo.loc[df_16_geo['gmmclusters'] == 2]
df_16_geo_3_gmm = df_16_geo.loc[df_16_geo['gmmclusters'] == 3]
```


```python
print ('mean value of median house price per each cluster:')
print (df_16_geo_0_gmm['mhval16'].mean())
print (df_16_geo_1_gmm['mhval16'].mean())
print (df_16_geo_2_gmm['mhval16'].mean())
print (df_16_geo_3_gmm['mhval16'].mean())
```

    mean value of median house price per each cluster:
    285860.8333333333
    553840.625
    509189.06605922553
    1258952.1739130435



```python
print ('mean value of percentage of college educated per each cluster:')
print (df_16_geo_0_gmm['per_col_16'].mean())
print (df_16_geo_1_gmm['per_col_16'].mean())
print (df_16_geo_2_gmm['per_col_16'].mean())
print (df_16_geo_3_gmm['per_col_16'].mean())
```

    mean value of percentage of college educated per each cluster:
    0.2879394686470429
    0.4285202473878268
    0.27995479783225713
    0.7178991994132167



```python
print ('mean value of share of very high income households per each cluster:')
print (df_16_geo_0_gmm['vhi2016'].mean())
print (df_16_geo_1_gmm['vhi2016'].mean())
print (df_16_geo_2_gmm['vhi2016'].mean())
print (df_16_geo_3_gmm['vhi2016'].mean())
```

    mean value of share of very high income households per each cluster:
    0.31794738170380393
    0.33505090029740875
    0.3462721257836147
    0.5138638045476831



```python
print ('mean value of share of median income households per each cluster:')
print (df_16_geo_0_gmm['hinc16'].mean())
print (df_16_geo_1_gmm['hinc16'].mean())
print (df_16_geo_2_gmm['hinc16'].mean())
print (df_16_geo_3_gmm['hinc16'].mean())
```

    mean value of share of median income households per each cluster:
    46190.34166666667
    65105.0
    56895.717539863326
    111193.69565217392



```python

fig = pl.figure(figsize=(50,10))

ax1 = fig.add_subplot(141)
ax1.set_title('2016 census(k-means)', fontsize=30)
#converting population to total
df_16_geo.plot(column='clusters',legend = True, cmap='viridis',categorical=True,ax=ax1)
ax1.axis('off')

ax2 = fig.add_subplot(142)
ax2.set_title('2016 census(gmm)', fontsize=30)
df_16_geo.plot(column='gmmclusters',legend = True,cmap='viridis',categorical=True, ax=ax2)
ax2.axis('off')


```




    (-74.28342262152397, -73.67222739857235, 40.47514405028754, 40.93624094984428)




![png](featureimportanceandclustering_files/featureimportanceandclustering_58_1.png)



```python
## The Color is randomly assigned, so dont use color to associae with cluster

```


```python
## Explaination: 
```

<a id='6'></a>
## 4.1. 2000 Data - K-means


```python
cols_00 = []
for column in data_census.columns:
    if '00' in column:
        cols_00.append(column)
cols_00 
print(len(cols_00),cols_00)
```

    26 ['per_col00', 'per_nonwhite00', 'per_rent00', 'vli2000', 'li2000', 'mi2000', 'hmi2000', 'hi2000', 'vhi2000', 'per_all_li00', 'hh00', 'all_li_count00', 'pop00', 'ch_all_li_count90_00', 'ch_all_li_count00_16', 'mrent00', 'rou00', 'mhval00', 'ohu00', 'mhval90_00', 'mhval00_16', 'mrent90_00', 'mrent00_16', 'hinc00', 'pct_ch_hinc90_00', 'pct_ch_hinc00_16']



```python
#create matrix with independent variable 
```


```python
df_00 = data_census[['per_col00', 'per_nonwhite00', 'per_rent00', 'vli2000', 'li2000', 'mi2000', 'hmi2000', 'hi2000', 'vhi2000', 'per_all_li00', 'hh00', 'all_li_count00', 'pop00', 'ch_all_li_count90_00', 'ch_all_li_count00_16', 'mrent00', 'rou00', 'mhval00', 'ohu00', 'mhval90_00', 'mhval00_16', 'mrent90_00', 'mrent00_16', 'hinc00', 'pct_ch_hinc90_00', 'pct_ch_hinc00_16']]
df_00.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>per_col00</th>
      <th>per_nonwhite00</th>
      <th>per_rent00</th>
      <th>vli2000</th>
      <th>li2000</th>
      <th>mi2000</th>
      <th>hmi2000</th>
      <th>hi2000</th>
      <th>vhi2000</th>
      <th>per_all_li00</th>
      <th>...</th>
      <th>rou00</th>
      <th>mhval00</th>
      <th>ohu00</th>
      <th>mhval90_00</th>
      <th>mhval00_16</th>
      <th>mrent90_00</th>
      <th>mrent00_16</th>
      <th>hinc00</th>
      <th>pct_ch_hinc90_00</th>
      <th>pct_ch_hinc00_16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.308353</td>
      <td>0.392846</td>
      <td>0.588881</td>
      <td>0.165321</td>
      <td>0.138535</td>
      <td>0.059451</td>
      <td>0.100951</td>
      <td>0.138998</td>
      <td>0.396744</td>
      <td>0.303856</td>
      <td>...</td>
      <td>858.000000</td>
      <td>230500.000000</td>
      <td>599.000000</td>
      <td>-0.010305</td>
      <td>1.419523</td>
      <td>0.371069</td>
      <td>0.772936</td>
      <td>77528.600000</td>
      <td>0.024456</td>
      <td>-0.217644</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.165272</td>
      <td>0.955867</td>
      <td>0.883148</td>
      <td>0.487993</td>
      <td>0.198152</td>
      <td>0.073064</td>
      <td>0.044690</td>
      <td>0.072732</td>
      <td>0.123368</td>
      <td>0.686145</td>
      <td>...</td>
      <td>1111.000000</td>
      <td>133982.312925</td>
      <td>147.000000</td>
      <td>-0.021966</td>
      <td>5.173949</td>
      <td>0.638764</td>
      <td>2.009309</td>
      <td>31193.126153</td>
      <td>0.338241</td>
      <td>1.463107</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.779420</td>
      <td>0.170020</td>
      <td>0.618152</td>
      <td>0.119530</td>
      <td>0.063569</td>
      <td>0.035175</td>
      <td>0.065627</td>
      <td>0.095087</td>
      <td>0.621012</td>
      <td>0.183099</td>
      <td>...</td>
      <td>1024.565796</td>
      <td>217699.995305</td>
      <td>632.898621</td>
      <td>-0.564601</td>
      <td>1.251723</td>
      <td>0.643761</td>
      <td>1.198020</td>
      <td>94268.847010</td>
      <td>0.237521</td>
      <td>0.069335</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.091283</td>
      <td>0.955941</td>
      <td>0.451968</td>
      <td>0.248304</td>
      <td>0.170377</td>
      <td>0.082061</td>
      <td>0.139400</td>
      <td>0.113894</td>
      <td>0.245964</td>
      <td>0.418681</td>
      <td>...</td>
      <td>598.304321</td>
      <td>192400.007635</td>
      <td>725.471985</td>
      <td>0.269967</td>
      <td>1.123700</td>
      <td>0.085799</td>
      <td>0.998638</td>
      <td>61377.049024</td>
      <td>-0.045143</td>
      <td>0.131579</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.124260</td>
      <td>0.993137</td>
      <td>0.651192</td>
      <td>0.389734</td>
      <td>0.147430</td>
      <td>0.087068</td>
      <td>0.033895</td>
      <td>0.064730</td>
      <td>0.277142</td>
      <td>0.537164</td>
      <td>...</td>
      <td>519.000000</td>
      <td>151700.000000</td>
      <td>278.000000</td>
      <td>0.348444</td>
      <td>1.257086</td>
      <td>-0.027027</td>
      <td>0.805556</td>
      <td>33757.450000</td>
      <td>-0.019158</td>
      <td>-0.148336</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
df_00= df_00.replace('nan',0)
df_00= df_00.replace(np.nan,0)
df_00= df_00.replace('NaN',0)
df_00.shape
```




    (2164, 26)




```python
corr = df_00.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr)
plt.yticks(rotation=0, size=7)
plt.xticks(rotation=90, size=7)
plt.show()
```


![png](featureimportanceandclustering_files/featureimportanceandclustering_66_0.png)



```python
rel_vars = corr.mhval00 [(corr.mhval00  > 0.2)]
rel_cols = list(rel_vars.index.values)

corr2 = df_00[rel_cols].corr()
plt.figure(figsize=(8,8))
hm = sns.heatmap(corr2, annot=True, annot_kws={'size':10})
plt.yticks(rotation=0, size=10)
plt.xticks(rotation=90, size=10)
plt.show()
```


![png](featureimportanceandclustering_files/featureimportanceandclustering_67_0.png)


## Same as 2016,  in 2000, the top three correlation variables are : per_col_16 --- "Percent college educated" /  mrent16 ---"Median house value" / vhi2016 --- "Share of very high income households".  (the college one is intresting, but the income one is kinda obvious i feel) 


```python
# Create a new dataframe with Geoid and geometry for variable from 2000
df_00_geo = data_census[['GEOID', 'geometry', 'per_col00', 'per_nonwhite00', 'per_rent00', 'vli2000', 'li2000', 'mi2000', 'hmi2000', 'hi2000', 'vhi2000', 'per_all_li00', 'hh00', 'all_li_count00', 'pop00', 'ch_all_li_count90_00', 'ch_all_li_count00_16', 'mrent00', 'rou00', 'mhval00', 'ohu00', 'mhval90_00', 'mhval00_16', 'mrent90_00', 'mrent00_16', 'hinc00', 'pct_ch_hinc90_00', 'pct_ch_hinc00_16']]
df_00_geo.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID</th>
      <th>geometry</th>
      <th>per_col00</th>
      <th>per_nonwhite00</th>
      <th>per_rent00</th>
      <th>vli2000</th>
      <th>li2000</th>
      <th>mi2000</th>
      <th>hmi2000</th>
      <th>hi2000</th>
      <th>...</th>
      <th>rou00</th>
      <th>mhval00</th>
      <th>ohu00</th>
      <th>mhval90_00</th>
      <th>mhval00_16</th>
      <th>mrent90_00</th>
      <th>mrent00_16</th>
      <th>hinc00</th>
      <th>pct_ch_hinc90_00</th>
      <th>pct_ch_hinc00_16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36081113900</td>
      <td>POLYGON ((-73.79190199967752 40.76893599959674...</td>
      <td>0.308353</td>
      <td>0.392846</td>
      <td>0.588881</td>
      <td>0.165321</td>
      <td>0.138535</td>
      <td>0.059451</td>
      <td>0.100951</td>
      <td>0.138998</td>
      <td>...</td>
      <td>858.000000</td>
      <td>230500.000000</td>
      <td>599.000000</td>
      <td>-0.010305</td>
      <td>1.419523</td>
      <td>0.371069</td>
      <td>0.772936</td>
      <td>77528.600000</td>
      <td>0.024456</td>
      <td>-0.217644</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36061025700</td>
      <td>POLYGON ((-73.95068000038171 40.81084300040413...</td>
      <td>0.165272</td>
      <td>0.955867</td>
      <td>0.883148</td>
      <td>0.487993</td>
      <td>0.198152</td>
      <td>0.073064</td>
      <td>0.044690</td>
      <td>0.072732</td>
      <td>...</td>
      <td>1111.000000</td>
      <td>133982.312925</td>
      <td>147.000000</td>
      <td>-0.021966</td>
      <td>5.173949</td>
      <td>0.638764</td>
      <td>2.009309</td>
      <td>31193.126153</td>
      <td>0.338241</td>
      <td>1.463107</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36047000502</td>
      <td>POLYGON ((-73.99136499969497 40.69701000031133...</td>
      <td>0.779420</td>
      <td>0.170020</td>
      <td>0.618152</td>
      <td>0.119530</td>
      <td>0.063569</td>
      <td>0.035175</td>
      <td>0.065627</td>
      <td>0.095087</td>
      <td>...</td>
      <td>1024.565796</td>
      <td>217699.995305</td>
      <td>632.898621</td>
      <td>-0.564601</td>
      <td>1.251723</td>
      <td>0.643761</td>
      <td>1.198020</td>
      <td>94268.847010</td>
      <td>0.237521</td>
      <td>0.069335</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36081015802</td>
      <td>POLYGON ((-73.81497399975029 40.68669500037402...</td>
      <td>0.091283</td>
      <td>0.955941</td>
      <td>0.451968</td>
      <td>0.248304</td>
      <td>0.170377</td>
      <td>0.082061</td>
      <td>0.139400</td>
      <td>0.113894</td>
      <td>...</td>
      <td>598.304321</td>
      <td>192400.007635</td>
      <td>725.471985</td>
      <td>0.269967</td>
      <td>1.123700</td>
      <td>0.085799</td>
      <td>0.998638</td>
      <td>61377.049024</td>
      <td>-0.045143</td>
      <td>0.131579</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36047116000</td>
      <td>POLYGON ((-73.88618300014529 40.66752100019399...</td>
      <td>0.124260</td>
      <td>0.993137</td>
      <td>0.651192</td>
      <td>0.389734</td>
      <td>0.147430</td>
      <td>0.087068</td>
      <td>0.033895</td>
      <td>0.064730</td>
      <td>...</td>
      <td>519.000000</td>
      <td>151700.000000</td>
      <td>278.000000</td>
      <td>0.348444</td>
      <td>1.257086</td>
      <td>-0.027027</td>
      <td>0.805556</td>
      <td>33757.450000</td>
      <td>-0.019158</td>
      <td>-0.148336</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
df_00_geo= df_00_geo.replace('nan',0)
df_00_geo= df_00_geo.replace(np.nan,0)
df_00_geo= df_00_geo.replace('NaN',0)
```


```python
# With 2000 data

print(df_00_geo.shape)

cols_X = [i for i in df_00_geo.columns if (i != 'mhval00')&(i != 'GEOID')&(i !='geometry')]
print(cols_X)
```

    (2164, 28)
    ['per_col00', 'per_nonwhite00', 'per_rent00', 'vli2000', 'li2000', 'mi2000', 'hmi2000', 'hi2000', 'vhi2000', 'per_all_li00', 'hh00', 'all_li_count00', 'pop00', 'ch_all_li_count90_00', 'ch_all_li_count00_16', 'mrent00', 'rou00', 'ohu00', 'mhval90_00', 'mhval00_16', 'mrent90_00', 'mrent00_16', 'hinc00', 'pct_ch_hinc90_00', 'pct_ch_hinc00_16']



```python
X_00 = np.asarray(df_00_geo[cols_X])
print(X.shape)
y_00 = np.asarray(df_00_geo['mhval00'])
y_00
```

    (2164, 33)





    array([230500.        , 133982.31292517, 217699.99530495, ...,
           625000.        , 175000.        , 361840.60721063])




```python
elbow(X_00, range(2,11))
```


![png](featureimportanceandclustering_files/featureimportanceandclustering_73_0.png)



![png](featureimportanceandclustering_files/featureimportanceandclustering_73_1.png)



```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

range_n_clusters = range(2,20)

for n_clusters in range_n_clusters:
    km = KMeans(n_clusters = n_clusters)
    res=km.fit(X_00)
    silhouette_avg = silhouette_score(X_00, res.labels_)
    print("For n_clusters = {},".format(n_clusters)+" the average silhouette_score is : {}".format(silhouette_avg))
    
```

    For n_clusters = 2, the average silhouette_score is : 0.5518688721634138
    For n_clusters = 3, the average silhouette_score is : 0.4976672614017573
    For n_clusters = 4, the average silhouette_score is : 0.5009178462198037
    For n_clusters = 5, the average silhouette_score is : 0.4989564273381826
    For n_clusters = 6, the average silhouette_score is : 0.5028497510841956
    For n_clusters = 7, the average silhouette_score is : 0.501302947634749
    For n_clusters = 8, the average silhouette_score is : 0.47518077252734264
    For n_clusters = 9, the average silhouette_score is : 0.4693093751222849
    For n_clusters = 10, the average silhouette_score is : 0.4840281238726348
    For n_clusters = 11, the average silhouette_score is : 0.46981351138897964
    For n_clusters = 12, the average silhouette_score is : 0.4665612142517066
    For n_clusters = 13, the average silhouette_score is : 0.45124467416992803
    For n_clusters = 14, the average silhouette_score is : 0.4455530799394592
    For n_clusters = 15, the average silhouette_score is : 0.42409773008431484
    For n_clusters = 16, the average silhouette_score is : 0.41825360813829865
    For n_clusters = 17, the average silhouette_score is : 0.4189437050802681
    For n_clusters = 18, the average silhouette_score is : 0.4090760225145433
    For n_clusters = 19, the average silhouette_score is : 0.4040941136659298


#### Choose 4 as cluster number 


```python
n=4 # number of clusters
dd=X_00 #data
tar=y_00 # real target

km=KMeans(n_clusters=n)
res=km.fit(dd)
```


```python
y_clusters = res.labels_
y_clusters
```




    array([1, 0, 1, ..., 0, 2, 1], dtype=int32)




```python
df_00_geo['clusters'] =y_clusters
df_00_geo['clusters'].unique()
df_00_geo.to_csv('clusters_00.csv')
df_00_geo.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID</th>
      <th>geometry</th>
      <th>per_col00</th>
      <th>per_nonwhite00</th>
      <th>per_rent00</th>
      <th>vli2000</th>
      <th>li2000</th>
      <th>mi2000</th>
      <th>hmi2000</th>
      <th>hi2000</th>
      <th>...</th>
      <th>mhval00</th>
      <th>ohu00</th>
      <th>mhval90_00</th>
      <th>mhval00_16</th>
      <th>mrent90_00</th>
      <th>mrent00_16</th>
      <th>hinc00</th>
      <th>pct_ch_hinc90_00</th>
      <th>pct_ch_hinc00_16</th>
      <th>clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2161</th>
      <td>36061021900</td>
      <td>POLYGON ((-73.9554510000636 40.81937700004859,...</td>
      <td>0.057858</td>
      <td>0.976335</td>
      <td>0.988011</td>
      <td>0.621887</td>
      <td>0.177527</td>
      <td>0.049810</td>
      <td>0.044513</td>
      <td>0.040544</td>
      <td>...</td>
      <td>625000.000000</td>
      <td>27.0</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.372990</td>
      <td>0.236534</td>
      <td>25044.400</td>
      <td>-0.002372</td>
      <td>-0.138131</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2162</th>
      <td>36061022102</td>
      <td>POLYGON ((-73.94515500010318 40.82087599992749...</td>
      <td>0.282637</td>
      <td>0.974572</td>
      <td>0.827789</td>
      <td>0.439632</td>
      <td>0.187200</td>
      <td>0.079572</td>
      <td>0.056022</td>
      <td>0.087187</td>
      <td>...</td>
      <td>175000.000000</td>
      <td>176.0</td>
      <td>-0.066667</td>
      <td>2.073714</td>
      <td>0.768997</td>
      <td>1.054983</td>
      <td>43768.250</td>
      <td>0.229670</td>
      <td>-0.048740</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2163</th>
      <td>36047152200</td>
      <td>POLYGON ((-73.96408699991511 40.6444069999831,...</td>
      <td>0.422212</td>
      <td>0.660112</td>
      <td>0.521779</td>
      <td>0.129393</td>
      <td>0.100985</td>
      <td>0.064595</td>
      <td>0.053479</td>
      <td>0.082005</td>
      <td>...</td>
      <td>361840.607211</td>
      <td>527.0</td>
      <td>0.205564</td>
      <td>2.168246</td>
      <td>0.398152</td>
      <td>0.976885</td>
      <td>84499.125</td>
      <td>-0.140776</td>
      <td>0.007667</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 29 columns</p>
</div>




```python
df_00_geo_new = df_00_geo[['GEOID','clusters']]
```


```python
df_00_geo_new.to_csv('clusters_00_k-means.csv')
```


```python
f, ax = plt.subplots(figsize=(10,10))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
df_00_geo.plot(column='clusters',cmap='viridis',alpha=1,
                  linewidth=0.1,ax=ax, categorical=True,legend=True, k=10)
plt.title("median property price clustering map for 2010(k-means)")
```




    Text(0.5, 1.0, 'median property price clustering map for 2010(k-means)')




![png](featureimportanceandclustering_files/featureimportanceandclustering_81_1.png)


<a id='7'></a>
## 4.2. 2000 Data - GaussianMixture


```python
gmm = GaussianMixture(n_components=4).fit(X_00)
labels_00 = gmm.predict(X_00)
```


```python
y_clusters = labels_00
y_clusters
```




    array([1, 0, 0, ..., 0, 0, 1])




```python
df_00_geo['gmmclusters'] =y_clusters
df_00_geo['gmmclusters'].unique()
df_00_geo.to_csv('clusters_00_gmm.csv')
df_00_geo.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID</th>
      <th>geometry</th>
      <th>per_col00</th>
      <th>per_nonwhite00</th>
      <th>per_rent00</th>
      <th>vli2000</th>
      <th>li2000</th>
      <th>mi2000</th>
      <th>hmi2000</th>
      <th>hi2000</th>
      <th>...</th>
      <th>ohu00</th>
      <th>mhval90_00</th>
      <th>mhval00_16</th>
      <th>mrent90_00</th>
      <th>mrent00_16</th>
      <th>hinc00</th>
      <th>pct_ch_hinc90_00</th>
      <th>pct_ch_hinc00_16</th>
      <th>clusters</th>
      <th>gmmclusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2161</th>
      <td>36061021900</td>
      <td>POLYGON ((-73.9554510000636 40.81937700004859,...</td>
      <td>0.057858</td>
      <td>0.976335</td>
      <td>0.988011</td>
      <td>0.621887</td>
      <td>0.177527</td>
      <td>0.049810</td>
      <td>0.044513</td>
      <td>0.040544</td>
      <td>...</td>
      <td>27.0</td>
      <td>0.000000</td>
      <td>-1.000000</td>
      <td>0.372990</td>
      <td>0.236534</td>
      <td>25044.400</td>
      <td>-0.002372</td>
      <td>-0.138131</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2162</th>
      <td>36061022102</td>
      <td>POLYGON ((-73.94515500010318 40.82087599992749...</td>
      <td>0.282637</td>
      <td>0.974572</td>
      <td>0.827789</td>
      <td>0.439632</td>
      <td>0.187200</td>
      <td>0.079572</td>
      <td>0.056022</td>
      <td>0.087187</td>
      <td>...</td>
      <td>176.0</td>
      <td>-0.066667</td>
      <td>2.073714</td>
      <td>0.768997</td>
      <td>1.054983</td>
      <td>43768.250</td>
      <td>0.229670</td>
      <td>-0.048740</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2163</th>
      <td>36047152200</td>
      <td>POLYGON ((-73.96408699991511 40.6444069999831,...</td>
      <td>0.422212</td>
      <td>0.660112</td>
      <td>0.521779</td>
      <td>0.129393</td>
      <td>0.100985</td>
      <td>0.064595</td>
      <td>0.053479</td>
      <td>0.082005</td>
      <td>...</td>
      <td>527.0</td>
      <td>0.205564</td>
      <td>2.168246</td>
      <td>0.398152</td>
      <td>0.976885</td>
      <td>84499.125</td>
      <td>-0.140776</td>
      <td>0.007667</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 30 columns</p>
</div>




```python
f, ax = plt.subplots(figsize=(10,10))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
df_00_geo.plot(column='gmmclusters',cmap='viridis',alpha=1,
                  linewidth=0.1,ax=ax, categorical=True,legend=True, k=10)
plt.title("median property price clustering map for 2010(gmm)")
```




    Text(0.5, 1.0, 'median property price clustering map for 2010(gmm)')




![png](featureimportanceandclustering_files/featureimportanceandclustering_86_1.png)



```python
# Check character for each cluster: 
```


```python
#K-means
df_00_geo_0 = df_00_geo.loc[df_00_geo['clusters'] == 0]
df_00_geo_1 = df_00_geo.loc[df_00_geo['clusters'] == 1]
df_00_geo_2 = df_00_geo.loc[df_00_geo['clusters'] == 2]
df_00_geo_3 = df_00_geo.loc[df_00_geo['clusters'] == 3]
```


```python
print ('mean value of median house price per each cluster:')
print (df_00_geo_0['mhval00'].mean())
print (df_00_geo_1['mhval00'].mean())
print (df_00_geo_2['mhval00'].mean())
print (df_00_geo_3['mhval00'].mean())
```

    mean value of median house price per each cluster:
    181691.12331504404
    263797.38707168703
    210767.24690402375
    518541.6406545601



```python
print ('mean value of percentage of college educated per each cluster:')
print (df_00_geo_0['per_col00'].mean())
print (df_00_geo_1['per_col00'].mean())
print (df_00_geo_2['per_col00'].mean())
print (df_00_geo_3['per_col00'].mean())
```

    mean value of percentage of college educated per each cluster:
    0.10035763466891287
    0.3720906185505747
    0.2149997156765612
    0.6856974332951582



```python
print ('mean value of median rent per each cluster:')
print (df_00_geo_0['mrent00'].mean())
print (df_00_geo_1['mrent00'].mean())
print (df_00_geo_2['mrent00'].mean())
print (df_00_geo_3['mrent00'].mean())
```

    mean value of median rent per each cluster:
    496.1151776103142
    907.5483845071093
    739.058735851456
    1397.0065717101697



```python
print ('mean value of median income per household per each cluster:')
print (df_00_geo_0['hinc00'].mean())
print (df_00_geo_1['hinc00'].mean())
print (df_00_geo_2['hinc00'].mean())
print (df_00_geo_3['hinc00'].mean())
```

    mean value of median income per household per each cluster:
    27663.847964840923
    85074.04151760307
    54374.57370006127
    146451.74128956822



```python
print ('mean value of median rent per each cluster:')
print (df_00_geo_0['mrent00'].mean())
print (df_00_geo_1['mrent00'].mean())
print (df_00_geo_2['mrent00'].mean())
print (df_00_geo_3['mrent00'].mean())
```

    mean value of median rent per each cluster:
    496.1151776103142
    907.5483845071093
    739.058735851456
    1397.0065717101697



```python
#GMM
df_00_geo_0_gmm = df_00_geo.loc[df_00_geo['gmmclusters'] == 0]
df_00_geo_1_gmm = df_00_geo.loc[df_00_geo['gmmclusters'] == 1]
df_00_geo_2_gmm = df_00_geo.loc[df_00_geo['gmmclusters'] == 2]
df_00_geo_3_gmm = df_00_geo.loc[df_00_geo['gmmclusters'] == 3]
```


```python
print ('mean value of median house price per each cluster:')
print (df_00_geo_0_gmm['mhval00'].mean())
print (df_00_geo_1_gmm['mhval00'].mean())
print (df_00_geo_2_gmm['mhval00'].mean())
print (df_00_geo_3_gmm['mhval00'].mean())
```

    mean value of median house price per each cluster:
    222156.90107853856
    226886.27320846854
    299973.46515160595
    116666.57747207336



```python
print ('mean value of percentage of college educated per each cluster:')
print (df_00_geo_0_gmm['per_col00'].mean())
print (df_00_geo_1_gmm['per_col00'].mean())
print (df_00_geo_2_gmm['per_col00'].mean())
print (df_00_geo_3_gmm['per_col00'].mean())
```

    mean value of percentage of college educated per each cluster:
    0.265287923377579
    0.20991558851353054
    0.3698865121521185
    0.13395070357780373



```python
print ('mean value of median rent per each cluster:')
print (df_00_geo_0_gmm['mrent00'].mean())
print (df_00_geo_1_gmm['mrent00'].mean())
print (df_00_geo_2_gmm['mrent00'].mean())
print (df_00_geo_3_gmm['mrent00'].mean())
```

    mean value of median rent per each cluster:
    692.923444812699
    768.7556997731809
    984.1447957513436
    324.098748186106



```python

fig = pl.figure(figsize=(50,10))

ax1 = fig.add_subplot(141)
ax1.set_title('2010 census', fontsize=30)
#converting population to total
df_00_geo.plot(column='clusters',legend = True, cmap='viridis',categorical=True,ax=ax1)
ax1.axis('off')

ax2 = fig.add_subplot(142)
ax2.set_title('2016 census(gmm)', fontsize=30)
df_00_geo.plot(column='gmmclusters',legend = True,cmap='viridis',categorical=True, ax=ax2)
ax2.axis('off')
```




    (-74.28342262152397, -73.67222739857235, 40.47514405028754, 40.93624094984428)




![png](featureimportanceandclustering_files/featureimportanceandclustering_98_1.png)





```python

```

<a id='8'></a>
## 5.1. 1990 Data - K-means


```python
cols_90 = []
for column in data_census.columns:
    if '90' in column:
        cols_90.append(column)
cols_90 
print(len(cols_90),cols_90 )
```

    21 ['per_nonwhite90', 'per_rent90', 'vli1990', 'li1990', 'mi1990', 'hmi1990', 'hi1990', 'vhi1990', 'per_all_li90', 'all_li_count90', 'ch_all_li_count90_00', 'mrent90', 'rou90', 'mhval90', 'ohu90', 'mhval90_00', 'mrent90_00', 'pop90', 'per_col90', 'hinc90', 'pct_ch_hinc90_00']



```python
df_90 = data_census[['per_nonwhite90', 'per_rent90', 'vli1990', 'li1990', 'mi1990', 'hmi1990', 'hi1990', 'vhi1990', 'per_all_li90', 'all_li_count90', 'ch_all_li_count90_00', 'mrent90', 'rou90', 'mhval90', 'ohu90', 'mhval90_00', 'mrent90_00', 'pop90', 'per_col90', 'hinc90', 'pct_ch_hinc90_00']]
df_90.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>per_nonwhite90</th>
      <th>per_rent90</th>
      <th>vli1990</th>
      <th>li1990</th>
      <th>mi1990</th>
      <th>hmi1990</th>
      <th>hi1990</th>
      <th>vhi1990</th>
      <th>per_all_li90</th>
      <th>all_li_count90</th>
      <th>...</th>
      <th>mrent90</th>
      <th>rou90</th>
      <th>mhval90</th>
      <th>ohu90</th>
      <th>mhval90_00</th>
      <th>mrent90_00</th>
      <th>pop90</th>
      <th>per_col90</th>
      <th>hinc90</th>
      <th>pct_ch_hinc90_00</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.204823</td>
      <td>0.548056</td>
      <td>0.151913</td>
      <td>0.183754</td>
      <td>0.056812</td>
      <td>0.100683</td>
      <td>0.121964</td>
      <td>0.384873</td>
      <td>0.335667</td>
      <td>467.920013</td>
      <td>...</td>
      <td>636.000003</td>
      <td>747.000000</td>
      <td>232900.001207</td>
      <td>616.000000</td>
      <td>-0.010305</td>
      <td>0.371069</td>
      <td>3193.000000</td>
      <td>0.250109</td>
      <td>75677.820392</td>
      <td>0.024456</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.825769</td>
      <td>1.789007</td>
      <td>1.076692</td>
      <td>0.247286</td>
      <td>0.159399</td>
      <td>0.100824</td>
      <td>0.194918</td>
      <td>0.220882</td>
      <td>1.323978</td>
      <td>1594.069214</td>
      <td>...</td>
      <td>329.308509</td>
      <td>1034.000000</td>
      <td>136991.523996</td>
      <td>118.000000</td>
      <td>-0.021966</td>
      <td>0.638764</td>
      <td>3241.000000</td>
      <td>0.333387</td>
      <td>23309.056491</td>
      <td>0.338241</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.071420</td>
      <td>0.287210</td>
      <td>0.067200</td>
      <td>0.035058</td>
      <td>0.031725</td>
      <td>0.025453</td>
      <td>0.047844</td>
      <td>0.237797</td>
      <td>0.102258</td>
      <td>163.299118</td>
      <td>...</td>
      <td>553.000011</td>
      <td>1030.796875</td>
      <td>500001.015257</td>
      <td>566.582214</td>
      <td>-0.564601</td>
      <td>0.643761</td>
      <td>2668.232910</td>
      <td>0.291718</td>
      <td>76175.578528</td>
      <td>0.237521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.502631</td>
      <td>0.252444</td>
      <td>0.090191</td>
      <td>0.115032</td>
      <td>0.070548</td>
      <td>0.027556</td>
      <td>0.077901</td>
      <td>0.178982</td>
      <td>0.205223</td>
      <td>239.708450</td>
      <td>...</td>
      <td>676.000009</td>
      <td>526.597412</td>
      <td>151500.002798</td>
      <td>642.000671</td>
      <td>0.269967</td>
      <td>0.085799</td>
      <td>4229.585449</td>
      <td>0.067395</td>
      <td>64278.752779</td>
      <td>-0.045143</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.994751</td>
      <td>0.839928</td>
      <td>0.386384</td>
      <td>0.162114</td>
      <td>0.122148</td>
      <td>0.090324</td>
      <td>0.050832</td>
      <td>0.188199</td>
      <td>0.548498</td>
      <td>329.098572</td>
      <td>...</td>
      <td>444.000000</td>
      <td>467.000000</td>
      <td>112500.000000</td>
      <td>89.000000</td>
      <td>0.348444</td>
      <td>-0.027027</td>
      <td>1905.000000</td>
      <td>0.092979</td>
      <td>34416.810000</td>
      <td>-0.019158</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
df_90= df_90.replace('nan',0)
df_90= df_90.replace(np.nan,0)
df_90= df_90.replace('NaN',0)
df_90.shape
```




    (2164, 21)




```python
corr = df_90.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr)
plt.yticks(rotation=0, size=7)
plt.xticks(rotation=90, size=7)
plt.show()
```


![png](featureimportanceandclustering_files/featureimportanceandclustering_105_0.png)



```python
rel_vars = corr.mhval90 [(corr.mhval90  > 0.2)]
rel_cols = list(rel_vars.index.values)

corr2 = df_90[rel_cols].corr()
plt.figure(figsize=(8,8))
hm = sns.heatmap(corr2, annot=True, annot_kws={'size':10})
plt.yticks(rotation=0, size=10)
plt.xticks(rotation=90, size=10)
plt.show()
```


![png](featureimportanceandclustering_files/featureimportanceandclustering_106_0.png)


### Evaluate the changes of cluster between 2000


```python
# Create a new dataframe with Geoid and geometry for variable from 2016
df_90_geo = data_census[['GEOID', 'geometry','per_nonwhite90', 'per_rent90', 'vli1990', 'li1990', 'mi1990', 'hmi1990', 'hi1990', 'vhi1990', 'per_all_li90', 'all_li_count90', 'ch_all_li_count90_00', 'mrent90', 'rou90', 'mhval90', 'ohu90', 'mhval90_00', 'mrent90_00', 'pop90', 'per_col90', 'hinc90', 'pct_ch_hinc90_00']]
```


```python
df_90_geo= df_90_geo.replace('nan',0)
df_90_geo= df_90_geo.replace(np.nan,0)
df_90_geo= df_90_geo.replace('NaN',0)
```


```python
print(df_90_geo.shape)

cols_X = [i for i in df_90_geo.columns if (i != 'mhval16')&(i != 'GEOID')&(i !='geometry')]
print(cols_X)
```

    (2164, 23)
    ['per_nonwhite90', 'per_rent90', 'vli1990', 'li1990', 'mi1990', 'hmi1990', 'hi1990', 'vhi1990', 'per_all_li90', 'all_li_count90', 'ch_all_li_count90_00', 'mrent90', 'rou90', 'mhval90', 'ohu90', 'mhval90_00', 'mrent90_00', 'pop90', 'per_col90', 'hinc90', 'pct_ch_hinc90_00']



```python
X_90 = np.asarray(df_90_geo[cols_X])
print(X.shape)
y_90= np.asarray(df_90_geo['mhval90'])
y_90
```

    (2164, 33)





    array([232900.00120653, 136991.5239962 , 500001.01525737, ...,
                0.        , 187500.        , 300142.25542915])




```python
elbow(X_90, range(2,11))
```


![png](featureimportanceandclustering_files/featureimportanceandclustering_112_0.png)



![png](featureimportanceandclustering_files/featureimportanceandclustering_112_1.png)



```python
range_n_clusters = range(2,20)

for n_clusters in range_n_clusters:
    km = KMeans(n_clusters = n_clusters)
    res=km.fit(X_90 )
    silhouette_avg = silhouette_score(X_90 , res.labels_)
    print("For n_clusters = {},".format(n_clusters)+" the average silhouette_score is : {}".format(silhouette_avg))
    
```

    For n_clusters = 2, the average silhouette_score is : 0.5127155146712002
    For n_clusters = 3, the average silhouette_score is : 0.6104681664267608
    For n_clusters = 4, the average silhouette_score is : 0.5033849339948169
    For n_clusters = 5, the average silhouette_score is : 0.4732202745681003
    For n_clusters = 6, the average silhouette_score is : 0.4751563393239955
    For n_clusters = 7, the average silhouette_score is : 0.4175895327430851
    For n_clusters = 8, the average silhouette_score is : 0.4153607659135086
    For n_clusters = 9, the average silhouette_score is : 0.40736318117170905
    For n_clusters = 10, the average silhouette_score is : 0.41476045822772534
    For n_clusters = 11, the average silhouette_score is : 0.40665242457165124
    For n_clusters = 12, the average silhouette_score is : 0.4045639495740838
    For n_clusters = 13, the average silhouette_score is : 0.39310563501612783
    For n_clusters = 14, the average silhouette_score is : 0.40120994878636074
    For n_clusters = 15, the average silhouette_score is : 0.3895355309368339
    For n_clusters = 16, the average silhouette_score is : 0.3949212908022343
    For n_clusters = 17, the average silhouette_score is : 0.3873145128838008
    For n_clusters = 18, the average silhouette_score is : 0.3888055496601713
    For n_clusters = 19, the average silhouette_score is : 0.3882340417677684



```python
n=4 # number of clusters
dd=X_90 #data
tar=y_90 # real target

km=KMeans(n_clusters=n)
res=km.fit(dd)
```


```python
y_clusters = res.labels_
y_clusters
```




    array([0, 3, 2, ..., 1, 3, 0], dtype=int32)




```python
df_90_geo['clusters'] =y_clusters
df_90_geo['clusters'].unique()
df_90_geo.to_csv('clusters_90.csv')
df_90_geo.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID</th>
      <th>geometry</th>
      <th>per_nonwhite90</th>
      <th>per_rent90</th>
      <th>vli1990</th>
      <th>li1990</th>
      <th>mi1990</th>
      <th>hmi1990</th>
      <th>hi1990</th>
      <th>vhi1990</th>
      <th>...</th>
      <th>rou90</th>
      <th>mhval90</th>
      <th>ohu90</th>
      <th>mhval90_00</th>
      <th>mrent90_00</th>
      <th>pop90</th>
      <th>per_col90</th>
      <th>hinc90</th>
      <th>pct_ch_hinc90_00</th>
      <th>clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2161</th>
      <td>36061021900</td>
      <td>POLYGON ((-73.9554510000636 40.81937700004859,...</td>
      <td>0.988397</td>
      <td>0.991701</td>
      <td>0.547147</td>
      <td>0.183213</td>
      <td>0.086845</td>
      <td>0.060155</td>
      <td>0.047117</td>
      <td>0.075524</td>
      <td>...</td>
      <td>1673.0</td>
      <td>0.000000</td>
      <td>14.0</td>
      <td>0.000000</td>
      <td>0.372990</td>
      <td>5085.0</td>
      <td>0.055690</td>
      <td>25103.939512</td>
      <td>-0.002372</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2162</th>
      <td>36061022102</td>
      <td>POLYGON ((-73.94515500010318 40.82087599992749...</td>
      <td>0.939631</td>
      <td>0.945455</td>
      <td>0.463633</td>
      <td>0.134022</td>
      <td>0.108665</td>
      <td>0.038336</td>
      <td>0.115069</td>
      <td>0.140275</td>
      <td>...</td>
      <td>780.0</td>
      <td>187500.000000</td>
      <td>45.0</td>
      <td>-0.066667</td>
      <td>0.768997</td>
      <td>1408.0</td>
      <td>0.161512</td>
      <td>35593.500000</td>
      <td>0.229670</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2163</th>
      <td>36047152200</td>
      <td>POLYGON ((-73.96408699991511 40.6444069999831,...</td>
      <td>0.862770</td>
      <td>0.865252</td>
      <td>0.230287</td>
      <td>0.074881</td>
      <td>0.081649</td>
      <td>0.144258</td>
      <td>0.191281</td>
      <td>1.277644</td>
      <td>...</td>
      <td>500.0</td>
      <td>300142.255429</td>
      <td>497.0</td>
      <td>0.205564</td>
      <td>0.398152</td>
      <td>3171.0</td>
      <td>1.036555</td>
      <td>98343.495042</td>
      <td>-0.140776</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 24 columns</p>
</div>




```python
# check cluster map 
f, ax = plt.subplots(figsize=(10,10))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
df_90_geo.plot(column='clusters',cmap='viridis',alpha=1,
                  linewidth=0.1,ax=ax, categorical=True,legend=True, k=10)
plt.title("median property price clustering map for 1990")
```




    Text(0.5, 1.0, 'median property price clustering map for 1990')




![png](featureimportanceandclustering_files/featureimportanceandclustering_117_1.png)


<a id='9'></a>
## 5.2. 1990 Data - GaussianMixture


```python
gmm = GaussianMixture(n_components=4).fit(X_90)
labels = gmm.predict(X_90)
```


```python
y_clusters = labels
y_clusters
```




    array([0, 2, 1, ..., 3, 3, 1])




```python
df_90_geo['gmmclusters'] =y_clusters
df_90_geo['gmmclusters'].unique()
df_90_geo.to_csv('clusters_90_gmm.csv')
df_90_geo.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID</th>
      <th>geometry</th>
      <th>per_nonwhite90</th>
      <th>per_rent90</th>
      <th>vli1990</th>
      <th>li1990</th>
      <th>mi1990</th>
      <th>hmi1990</th>
      <th>hi1990</th>
      <th>vhi1990</th>
      <th>...</th>
      <th>mhval90</th>
      <th>ohu90</th>
      <th>mhval90_00</th>
      <th>mrent90_00</th>
      <th>pop90</th>
      <th>per_col90</th>
      <th>hinc90</th>
      <th>pct_ch_hinc90_00</th>
      <th>clusters</th>
      <th>gmmclusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2161</th>
      <td>36061021900</td>
      <td>POLYGON ((-73.9554510000636 40.81937700004859,...</td>
      <td>0.988397</td>
      <td>0.991701</td>
      <td>0.547147</td>
      <td>0.183213</td>
      <td>0.086845</td>
      <td>0.060155</td>
      <td>0.047117</td>
      <td>0.075524</td>
      <td>...</td>
      <td>0.000000</td>
      <td>14.0</td>
      <td>0.000000</td>
      <td>0.372990</td>
      <td>5085.0</td>
      <td>0.055690</td>
      <td>25103.939512</td>
      <td>-0.002372</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2162</th>
      <td>36061022102</td>
      <td>POLYGON ((-73.94515500010318 40.82087599992749...</td>
      <td>0.939631</td>
      <td>0.945455</td>
      <td>0.463633</td>
      <td>0.134022</td>
      <td>0.108665</td>
      <td>0.038336</td>
      <td>0.115069</td>
      <td>0.140275</td>
      <td>...</td>
      <td>187500.000000</td>
      <td>45.0</td>
      <td>-0.066667</td>
      <td>0.768997</td>
      <td>1408.0</td>
      <td>0.161512</td>
      <td>35593.500000</td>
      <td>0.229670</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2163</th>
      <td>36047152200</td>
      <td>POLYGON ((-73.96408699991511 40.6444069999831,...</td>
      <td>0.862770</td>
      <td>0.865252</td>
      <td>0.230287</td>
      <td>0.074881</td>
      <td>0.081649</td>
      <td>0.144258</td>
      <td>0.191281</td>
      <td>1.277644</td>
      <td>...</td>
      <td>300142.255429</td>
      <td>497.0</td>
      <td>0.205564</td>
      <td>0.398152</td>
      <td>3171.0</td>
      <td>1.036555</td>
      <td>98343.495042</td>
      <td>-0.140776</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 25 columns</p>
</div>




```python
f, ax = plt.subplots(figsize=(10,10))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
df_90_geo.plot(column='gmmclusters',cmap='viridis',alpha=1,
                  linewidth=0.1,ax=ax, categorical=True,legend=True, k=10)
plt.title("median property price clustering map for 1990(gmm)")
```




    Text(0.5, 1.0, 'median property price clustering map for 1990(gmm)')




![png](featureimportanceandclustering_files/featureimportanceandclustering_122_1.png)



```python
## Check each cluster's characters
```


```python
#K-means: 
df_90_geo_0 = df_90_geo.loc[df_90_geo['clusters'] == 0]
df_90_geo_1 = df_90_geo.loc[df_90_geo['clusters'] == 1]
df_90_geo_2 = df_90_geo.loc[df_90_geo['clusters'] == 2]
df_90_geo_3 = df_90_geo.loc[df_90_geo['clusters'] == 3]
```


```python
print ('mean value of median house price per each cluster:')
print (df_90_geo_0['mhval90'].mean())
print (df_90_geo_1['mhval90'].mean())
print (df_90_geo_2['mhval90'].mean())
print (df_90_geo_3['mhval90'].mean())
```

    mean value of median house price per each cluster:
    241939.85553957114
    10310.6506633012
    462719.54723601573
    154477.11482040564



```python
print ('mean value of median income per household per each cluster:')
print (df_90_geo_0['hinc90'].mean())
print (df_90_geo_1['hinc90'].mean())
print (df_90_geo_2['hinc90'].mean())
print (df_90_geo_3['hinc90'].mean())
```

    mean value of median income per household per each cluster:
    64598.2291641529
    36646.437830059294
    89627.58639196554
    56513.04784481334



```python
# check mean value 
print ('mean value of percentage of college educated per each cluster:')
print (df_90_geo_0['per_col90'].mean())
print (df_90_geo_1['per_col90'].mean())
print (df_90_geo_2['per_col90'].mean())
print (df_90_geo_3['per_col90'].mean())
```

    mean value of percentage of college educated per each cluster:
    0.23616383314608883
    0.16858069308635978
    0.5642394525731409
    0.1394356209001479



```python
print ('mean value of Owner-occupied housing units per each cluster:')
print (df_90_geo_0['ohu90'].mean())
print (df_90_geo_1['ohu90'].mean())
print (df_90_geo_2['ohu90'].mean())
print (df_90_geo_3['ohu90'].mean())
```

    mean value of Owner-occupied housing units per each cluster:
    482.25882837339935
    149.58570446968275
    764.2167039853831
    358.5511023471266



```python
print ('mean value of Owner-occupied housing units per each cluster:')
print (df_90_geo_0['vhi1990'].mean())
print (df_90_geo_1['vhi1990'].mean())
print (df_90_geo_2['vhi1990'].mean())
print (df_90_geo_3['vhi1990'].mean())
```

    mean value of Owner-occupied housing units per each cluster:
    0.3706484141886724
    0.19126921306368386
    0.5355311800976779
    0.32623628990614567



```python
#GMM: 
```


```python
df_90_geo_0_gmm = df_90_geo.loc[df_90_geo['gmmclusters'] == 0]
df_90_geo_1_gmm = df_90_geo.loc[df_90_geo['gmmclusters'] == 1]
df_90_geo_2_gmm = df_90_geo.loc[df_90_geo['gmmclusters'] == 2]
df_90_geo_3_gmm = df_90_geo.loc[df_90_geo['gmmclusters'] == 3]
```


```python
print ('mean value of median house price per each cluster:')
print (df_90_geo_0_gmm['mhval90'].mean())
print (df_90_geo_1_gmm['mhval90'].mean())
print (df_90_geo_2_gmm['mhval90'].mean())
print (df_90_geo_3_gmm['mhval90'].mean())
```

    mean value of median house price per each cluster:
    193806.43723173605
    210875.22853753128
    70442.24639729146
    98315.63636448285



```python
print ('mean value of median income per household per each cluster:')
print (df_90_geo_0_gmm['hinc90'].mean())
print (df_90_geo_1_gmm['hinc90'].mean())
print (df_90_geo_2_gmm['hinc90'].mean())
print (df_90_geo_3_gmm['hinc90'].mean())
```

    mean value of median income per household per each cluster:
    61815.19375100457
    73803.2937280715
    45331.41149754632
    34546.89954432665



```python
# check mean value 
print ('mean value of percentage of college educated per each cluster:')
print (df_90_geo_0_gmm['per_col90'].mean())
print (df_90_geo_1_gmm['per_col90'].mean())
print (df_90_geo_2_gmm['per_col90'].mean())
print (df_90_geo_3_gmm['per_col90'].mean())
```

    mean value of percentage of college educated per each cluster:
    0.17292143528319862
    0.3633068495448919
    0.20984752261426312
    0.10311695135616823



```python
print ('mean value of Owner-occupied housing units per each cluster:')
print (df_90_geo_0_gmm['ohu90'].mean())
print (df_90_geo_1_gmm['ohu90'].mean())
print (df_90_geo_2_gmm['ohu90'].mean())
print (df_90_geo_3_gmm['ohu90'].mean())
```

    mean value of Owner-occupied housing units per each cluster:
    387.84521206191215
    663.655358874228
    189.84912476563372
    146.90479883360118



```python
print ('mean value of Owner-occupied housing units per each cluster:')
print (df_90_geo_0_gmm['vhi1990'].mean())
print (df_90_geo_1_gmm['vhi1990'].mean())
print (df_90_geo_2_gmm['vhi1990'].mean())
print (df_90_geo_3_gmm['vhi1990'].mean())
```

    mean value of Owner-occupied housing units per each cluster:
    0.3582404721098998
    0.40393929473427675
    0.3827092759559495
    0.17453330464715713



```python

fig = pl.figure(figsize=(50,10))

ax1 = fig.add_subplot(141)
ax1.set_title('1990 census', fontsize=30)
#converting population to total
df_90_geo.plot(column='clusters',legend = True, cmap='viridis',categorical=True,ax=ax1)
ax1.axis('off')

ax2 = fig.add_subplot(142)
ax2.set_title('1990 census(gmm)', fontsize=30)
df_90_geo.plot(column='gmmclusters',legend = True,cmap='viridis',categorical=True, ax=ax2)
ax2.axis('off')
```




    (-74.28342262152397, -73.67222739857235, 40.47514405028754, 40.93624094984428)




![png](featureimportanceandclustering_files/featureimportanceandclustering_137_1.png)





```python

```

<a id='10'></a>
## 6. Cluster Difference


```python

fig = pl.figure(figsize=(50,10))

ax1 = fig.add_subplot(141)
ax1.set_title('2016 census', fontsize=30)
#converting population to total
df_16_geo.plot(column='clusters',legend = True, ax=ax1)
ax1.axis('off')

ax2 = fig.add_subplot(142)
ax2.set_title('2000 census', fontsize=30)
df_00_geo.plot(column='clusters',legend = True, ax=ax2)
ax2.axis('off')

ax3 = fig.add_subplot(143)
ax3.set_title('1990 census', fontsize=30)
df_90_geo.plot(column='clusters',legend = True, ax=ax3)
ax3.axis('off')


```




    (-74.28342262152395, -73.67222739857235, 40.47514405028754, 40.93624094984428)




![png](featureimportanceandclustering_files/featureimportanceandclustering_141_1.png)



```python

```

<a id='11'></a>
## 7. Identify the changes of each variables from 2000 to 2016

## Because census from 1990 has too much missing data. Lets check the changes of different variable in those clusers from 2000 to 2016. We aim to find the area that changes the most.
### We will calculate the difference between 2016 and 2000 of different variable. The dataset we are using already have the calculated variable. we can directly use. 


```python
data_census.columns
```




    Index(['GEOID', 'geometry', 'hh16', 'moe_hh16', 'per_col00', 'per_nonwhite00',
           'per_rent00', 'vli2000', 'li2000', 'mi2000', 'hmi2000', 'hi2000',
           'vhi2000', 'per_all_li00', 'hh00', 'all_li_count00', 'per_nonwhite90',
           'per_rent90', 'vli1990', 'li1990', 'mi1990', 'hmi1990', 'hi1990',
           'vhi1990', 'denominator', 'per_all_li90', 'all_li_count90', 'pop00',
           'pop16', 'moe_pop16', 'ch_all_li_count90_00', 'ch_all_li_count00_16',
           'moveinincd16', 'co_med_indinc16', 'per_limove16', 'per_limove09',
           'TOD', 'empd15', 'mrent90', 'rou90', 'mrent00', 'rou00', 'mrent16',
           'moe_mrent16', 'mhval90', 'ohu90', 'mhval00', 'ohu00', 'mhval16',
           'moe_mhval16', 'mhval90_00', 'mhval00_16', 'mrent90_00', 'mrent00_16',
           'hu16', 'per_units_pre50', 'ohu16', 'moe_owner16', 'rhu16',
           'moe_renter16', 'per_rent16', 'per_nonwhite16', 'vli2016', 'li2016',
           'mi2016', 'hmi2016', 'hi2016', 'vhi2016', 'per_all_li16',
           'all_li_count16', 'ag25up', 'bachelors', 'moe_bachelors', 'masters',
           'moe_masters', 'professional', 'moe_professionals', 'doctorate',
           'moe_doctorates', 'per_col_16', 'pop90', 'per_col90', 'moe_hu16',
           'hinc16', 'moe_hinc16', 'popgrowth', 'hinc00', 'hinc90',
           'pct_ch_hinc90_00', 'pct_ch_hinc00_16'],
          dtype='object')




```python
df_16_geo.columns
```




    Index(['GEOID', 'geometry', 'hh16', 'moe_hh16', 'pop16', 'moe_pop16',
           'ch_all_li_count00_16', 'moveinincd16', 'co_med_indinc16',
           'per_limove16', 'mrent16', 'moe_mrent16', 'mhval16', 'moe_mhval16',
           'mhval00_16', 'mrent00_16', 'hu16', 'ohu16', 'moe_owner16', 'rhu16',
           'moe_renter16', 'per_rent16', 'per_nonwhite16', 'vli2016', 'li2016',
           'mi2016', 'hmi2016', 'hi2016', 'vhi2016', 'per_all_li16',
           'all_li_count16', 'per_col_16', 'moe_hu16', 'hinc16', 'moe_hinc16',
           'pct_ch_hinc00_16', 'clusters'],
          dtype='object')




```python
# Change of college population
data_census['ch_col'] = ((data_census['per_col_16']* data_census['pop16'])- \
                            (data_census['per_col00']* data_census['pop00']))

data_census['ch_nonwhite'] = ((data_census['per_nonwhite16']* data_census['pop16'])- \
                            (data_census['per_nonwhite00']* data_census['pop00'])) 


```


```python
# Create variable of changes from 2000 to 2016:
data_census['ch_hh']=data_census['hh16']-data_census['hh00'] #change of number of household 
data_census['ch_pop'] = data_census['pop16']-data_census['pop00'] #change of number of household
 # change on non-white population
data_census['ch_ohu'] =  data_census['ohu16']-data_census['ohu00'] # Change of Owner-occupied housing units
data_census['ch_rhu'] = data_census['rhu16']-data_census['rou00'] #Changes of Renter-occupied housing units

data_census['ch_mhval'] = data_census['mhval16']-data_census['mhval00'] # change of median house price
data_census['ch_mrent'] = data_census['mrent16']-data_census['mrent00'] # change of median rent 
```


```python
cols_ch = []
for column in data_census.columns:
    if 'ch_' in column:
        cols_ch.append(column)
cols_ch 
print(len(cols_ch),cols_ch )
```

    12 ['ch_all_li_count90_00', 'ch_all_li_count00_16', 'pct_ch_hinc90_00', 'pct_ch_hinc00_16', 'ch_col', 'ch_nonwhite', 'ch_hh', 'ch_pop', 'ch_ohu', 'ch_rhu', 'ch_mhval', 'ch_mrent']



```python
df_ch = data_census[['GEOID', 'geometry','ch_all_li_count90_00', 'ch_all_li_count00_16', 'pct_ch_hinc90_00', 'pct_ch_hinc00_16', 'ch_hh', 'ch_pop', 'ch_nonwhite', 'ch_ohu', 'ch_rhu', 'ch_col', 'ch_mhval', 'ch_mrent']]
df_ch.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID</th>
      <th>geometry</th>
      <th>ch_all_li_count90_00</th>
      <th>ch_all_li_count00_16</th>
      <th>pct_ch_hinc90_00</th>
      <th>pct_ch_hinc00_16</th>
      <th>ch_hh</th>
      <th>ch_pop</th>
      <th>ch_nonwhite</th>
      <th>ch_ohu</th>
      <th>ch_rhu</th>
      <th>ch_col</th>
      <th>ch_mhval</th>
      <th>ch_mrent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36081113900</td>
      <td>POLYGON ((-73.79190199967752 40.76893599959674...</td>
      <td>-25.201660</td>
      <td>133.823456</td>
      <td>0.024456</td>
      <td>-0.217644</td>
      <td>-88.000000</td>
      <td>-117.000000</td>
      <td>690.000065</td>
      <td>10.000000</td>
      <td>-98.000000</td>
      <td>42.726673</td>
      <td>327200.000000</td>
      <td>674.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36061025700</td>
      <td>POLYGON ((-73.95068000038171 40.81084300040413...</td>
      <td>-730.898315</td>
      <td>-126.051941</td>
      <td>0.338241</td>
      <td>1.463107</td>
      <td>545.000000</td>
      <td>1588.000000</td>
      <td>397.000122</td>
      <td>447.000000</td>
      <td>98.000000</td>
      <td>2412.427383</td>
      <td>693217.687075</td>
      <td>1084.341134</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36047000502</td>
      <td>POLYGON ((-73.99136499969497 40.69701000031133...</td>
      <td>140.181107</td>
      <td>7.001373</td>
      <td>0.237521</td>
      <td>0.069335</td>
      <td>-129.464478</td>
      <td>-80.943115</td>
      <td>155.126230</td>
      <td>-120.898621</td>
      <td>-8.565796</td>
      <td>163.806271</td>
      <td>272500.004695</td>
      <td>1088.999980</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36081015802</td>
      <td>POLYGON ((-73.81497399975029 40.68669500037402...</td>
      <td>314.531799</td>
      <td>-100.156677</td>
      <td>-0.045143</td>
      <td>0.131579</td>
      <td>-88.776367</td>
      <td>-46.000000</td>
      <td>-105.758757</td>
      <td>-10.471985</td>
      <td>-78.304321</td>
      <td>470.039613</td>
      <td>216199.992365</td>
      <td>732.999986</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36047116000</td>
      <td>POLYGON ((-73.88618300014529 40.66752100019399...</td>
      <td>99.021484</td>
      <td>124.415955</td>
      <td>-0.019158</td>
      <td>-0.148336</td>
      <td>99.000000</td>
      <td>-446.000000</td>
      <td>-429.000059</td>
      <td>-13.000000</td>
      <td>112.000000</td>
      <td>-13.908952</td>
      <td>190700.000000</td>
      <td>348.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Use another clustering method Gaussian Mixture Models (EM)
```


```python
df_ch= df_ch.replace('nan',0)
df_ch= df_ch.replace(np.nan,0)
df_ch= df_ch.replace('NaN',0)
df_ch.shape
```




    (2164, 14)




```python
# check cluster map for median price change 
f, ax = plt.subplots(figsize=(10,10))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
df_ch.plot(column='ch_mhval',cmap='viridis',alpha=1,
                  linewidth=0.1,ax=ax, scheme="Equal_interval",legend=True, k=10)
plt.title("median property price changes for 2000 -2016 changes")
```




    Text(0.5, 1.0, 'median property price changes for 2000 -2016 changes')




![png](featureimportanceandclustering_files/featureimportanceandclustering_153_1.png)



```python
#  map for college population change 
f, ax = plt.subplots(figsize=(10,10))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
df_ch.plot(column='ch_col',cmap='viridis',alpha=1,
                  linewidth=0.1,ax=ax, scheme="Equal_interval",legend=True, k=10)
plt.title("college population changes from 2000 - 2016 changes")
```




    Text(0.5, 1.0, 'college population changes from 2000 - 2016 changes')




![png](featureimportanceandclustering_files/featureimportanceandclustering_154_1.png)



```python
# map for non_white population change
f, ax = plt.subplots(figsize=(10,10))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
df_ch.plot(column='ch_nonwhite',cmap='viridis',alpha=1,linewidth=0.2,
                  ax=ax, scheme="Equal_interval",legend=True, k=10)
plt.title("non-white population changes from 2000 -2016 changes")
```




    Text(0.5, 1.0, 'non-white population changes from 2000 -2016 changes')




![png](featureimportanceandclustering_files/featureimportanceandclustering_155_1.png)



```python
#  map for  population change
f, ax = plt.subplots(figsize=(10,10))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
df_ch.plot(column='ch_pop',cmap='viridis',alpha=1,
                  linewidth=0.1,ax=ax, scheme="Equal_interval",legend=True, k=10)
plt.title("population changes from 2000 -2016 changes")
```




    Text(0.5, 1.0, 'population changes from 2000 -2016 changes')




![png](featureimportanceandclustering_files/featureimportanceandclustering_156_1.png)



```python

```


```python
range_n_clusters = range(2,20)

for n_clusters in range_n_clusters:
    km = KMeans(n_clusters = n_clusters)
    res=km.fit(X_ch)
    silhouette_avg = silhouette_score(X_ch, res.labels_)
    print("For n_clusters = {},".format(n_clusters)+" the average silhouette_score is : {}".format(silhouette_avg))
    
        
```

    For n_clusters = 2, the average silhouette_score is : 0.3994190209626735
    For n_clusters = 3, the average silhouette_score is : 0.2700123153746197
    For n_clusters = 4, the average silhouette_score is : 0.2829920171319272
    For n_clusters = 5, the average silhouette_score is : 0.2114432391098041
    For n_clusters = 6, the average silhouette_score is : 0.22202456900833503
    For n_clusters = 7, the average silhouette_score is : 0.22050457191472292
    For n_clusters = 8, the average silhouette_score is : 0.1912177254557021
    For n_clusters = 9, the average silhouette_score is : 0.19221763990335464
    For n_clusters = 10, the average silhouette_score is : 0.17489717399506094
    For n_clusters = 11, the average silhouette_score is : 0.187970888282473
    For n_clusters = 12, the average silhouette_score is : 0.18831731024557016
    For n_clusters = 13, the average silhouette_score is : 0.19244596184759563
    For n_clusters = 14, the average silhouette_score is : 0.17846091592890062
    For n_clusters = 15, the average silhouette_score is : 0.18008386250989566
    For n_clusters = 16, the average silhouette_score is : 0.15585482681131627
    For n_clusters = 17, the average silhouette_score is : 0.14805356669718991
    For n_clusters = 18, the average silhouette_score is : 0.1530790913620238
    For n_clusters = 19, the average silhouette_score is : 0.15407018959700955



```python
n=4 # number of clusters
dd=X_ch #data
tar=y_ch # real target

km=KMeans(n_clusters=n)
res=km.fit(dd)
```


```python
y_clusters = res.labels_
y_clusters
```




    array([3, 0, 3, ..., 3, 3, 3], dtype=int32)




```python
df_ch['clusters'] =y_clusters
df_ch['clusters'].unique()
df_ch.to_csv('clusters_ch.csv')
df_ch.tail(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEOID</th>
      <th>geometry</th>
      <th>ch_all_li_count90_00</th>
      <th>ch_all_li_count00_16</th>
      <th>pct_ch_hinc90_00</th>
      <th>pct_ch_hinc00_16</th>
      <th>ch_hh</th>
      <th>ch_pop</th>
      <th>ch_nonwhite</th>
      <th>ch_ohu</th>
      <th>ch_rhu</th>
      <th>ch_col</th>
      <th>ch_mhval</th>
      <th>ch_mrent</th>
      <th>clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2161</th>
      <td>36061021900</td>
      <td>POLYGON ((-73.9554510000636 40.81937700004859,...</td>
      <td>514.117920</td>
      <td>-57.557861</td>
      <td>-0.002372</td>
      <td>-0.138131</td>
      <td>-169.0</td>
      <td>-8.0</td>
      <td>-153.999831</td>
      <td>31.0</td>
      <td>-200.0</td>
      <td>823.068672</td>
      <td>-625000.000000</td>
      <td>101.000000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2162</th>
      <td>36061022102</td>
      <td>POLYGON ((-73.94515500010318 40.82087599992749...</td>
      <td>194.173767</td>
      <td>22.305359</td>
      <td>0.229670</td>
      <td>-0.048740</td>
      <td>95.0</td>
      <td>288.0</td>
      <td>-219.000053</td>
      <td>94.0</td>
      <td>1.0</td>
      <td>662.414649</td>
      <td>362900.000000</td>
      <td>614.000000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2163</th>
      <td>36047152200</td>
      <td>POLYGON ((-73.96408699991511 40.6444069999831,...</td>
      <td>-76.315445</td>
      <td>30.123215</td>
      <td>-0.140776</td>
      <td>0.007667</td>
      <td>-38.0</td>
      <td>89.0</td>
      <td>-202.000138</td>
      <td>107.0</td>
      <td>-145.0</td>
      <td>525.806790</td>
      <td>784559.392789</td>
      <td>757.043478</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check cluster map 
f, ax = plt.subplots(figsize=(10,10))
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
df_ch.plot(column='clusters',cmap='viridis',alpha=1,
                  linewidth=0.1,ax=ax, categorical=True,legend=True, k=10)
plt.title("median property price clustering map for 2000 -2016 changes")
```




    Text(0.5, 1.0, 'median property price clustering map for 2000 -2016 changes')




![png](featureimportanceandclustering_files/featureimportanceandclustering_162_1.png)



```python

```

<a id='12'></a>
## 8. Summary and Analysis

#### Feature Importance of three years are similar: 
#### After check the variable dictionary, the top three correlation variables are : per_col_16 --- "Percent college educated" /  mrent16 ---"Median house value" / vhi2016 --- "Share of very high income households". 

#### For clusters in 2016: 
#### based on the map and mean value of those higher correlated variables, we can find out that. Cluster 1: Include Hudson Yard - Chelsea-Flatiron-Union Square/ Parts of Upper West and East Side/ parts of downtown brooklyn/ little part of Willimasburg. These areas have highest percentage of high income population, and highest percentage of college education and median household income. On the contrary, cluster 0 has the lowest mean value for all the four variables. From the clustering map above, the spatial distribution of cluster 0 are mainly in inner Queens and Brooklyn. 


#### For clusters in 2000: 
#### The areas in Cluster 0 represent the highest median property value, percentage of college education, and highest median income. The areas include: lower manhattan, east village, upper east side. and Douglas Manor/ Jamaica Estates-Holliswood( theres areas are not anymore at "highest" cluster in 2016). And Cluster 3 has lowest variable values. Area includes: areas above Harlem/ inner areas of Queens and brooklyn. (Williansburge in 2000 is still in "lowest" cluster) 

#### For cluster in 1990: 
#### In 1990, The cluster has highest value (cluster0) and cluster has lowest value (cluster 2) are both located in Manhattan. The "highest" areas are still in upper East/West side and lower manhattan area; however the other area is lowest part. (One of the reason might be 1990 has more missing data than other two years) 

#### Summary from the previous task, variables that have higher correlation with median property value ("mhval16/00/19") are: (list based on ranking of correlation). 2016:  "Percent college educated" / "Share of very high income households"/"Median rent value". 2000: "Percent college educated"//"Median rent value"/ Median Household income" / "Share of very high income households". 1990: Median Household income"/ "Percent college educated"/"Owner-occupied housing units"/"Share of very high income households"/



```python

```


```python

```


```python

```
