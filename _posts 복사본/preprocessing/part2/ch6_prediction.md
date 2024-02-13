---
title: "6. prediction"
category: "Datapreporcessing"
date: "2022-05-06"
thumbnail: "./data/preprocessing.png"
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

os.getcwd()
```

    'd:\\git_local_repository\\yangoos57\\ML\\Hands_On_Data_preprocessing_in_python\\Part2'

```python
msu_df = pd.read_csv('data/ch6/MSU applications.csv', index_col='Year')
msu_df.head(1)
msu_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 16 entries, 2006 to 2021
    Data columns (total 3 columns):
     #   Column                  Non-Null Count  Dtype
    ---  ------                  --------------  -----
     0   P_Football_Performance  16 non-null     float64
     1   SMAn2                   16 non-null     float64
     2   N_Applications          16 non-null     int64
    dtypes: float64(2), int64(1)
    memory usage: 512.0 bytes

### Regression

- Predictor attribute = dependent attribute = X axis
- target attribute = independent attribute = Y axis

<br>

sklearn 명령어

- .intercept\_ = B_0을 보여주는 값
- .coef\_ = B_1,B_2를 보여주는 값

```python
x = ['P_Football_Performance','SMAn2']
y ='N_Applications'

data_x = msu_df[x]
data_y = msu_df[y]

lm = LinearRegression()
lm.fit(data_x, data_y)

print('intercept (b0)', lm.intercept_)
coef_names = ['b1','b2']
pd.DataFrame({'Predictor': data_x.columns,
                    'coefficient Name': coef_names,
                    'coefficient Value' : lm.coef_})
```

    intercept (b0) -890.7106225983425

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
      <th>Predictor</th>
      <th>coefficient Name</th>
      <th>coefficient Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>P_Football_Performance</td>
      <td>b1</td>
      <td>5544.961933</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SMAn2</td>
      <td>b2</td>
      <td>0.907032</td>
    </tr>
  </tbody>
</table>
</div>

### scikit-learn으로 예측하기

x값

- 22년도 P_Football_Performance : 0.364
- 20,21년도 지원자 평균 : 17198

y값 예측

- .predict()

```python
newData = pd.DataFrame({'P_Football_Performance':0.364, 'SMAn2':17198}, index=[2022])

lm.predict(newData)
```

    array([16726.78787061])

### NLP

Heuristic approach

the most famous heuristic that is used to estimate the connections' weights for MLP is called backpropagation

**Backpropagation**

1. Random initialization : each connection's weight is first assigned a random number between -1 and 1.
2. Every time a data object is exposed to the MLP network, MLP xpects its dependent attribute.
3. epoch of learning : Every time all the data objets are exposed to the network.

```python
from sklearn.neural_network import MLPRegressor

x = ['P_Football_Performance','SMAn2']
y ='N_Applications'

data_x = msu_df[x]
data_y = msu_df[y]

newData = pd.DataFrame({'P_Football_Performance':0.364, 'SMAn2':17198}, index=[2022])

mlp = MLPRegressor(hidden_layer_sizes=6, max_iter=100000)
mlp.fit(data_x, data_y)
mlp.predict(newData)
```

    array([18641.62328469])

```python

```
