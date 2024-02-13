---
title: "13. Data Reduction"
category: "Datapreporcessing"
date: "2022-05-13"
thumbnail: "./data/preprocessing.png"
---

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

os.getcwd()
```

    'd:\\git_local_repository\\yangoos57\\ML\\Hands_On_Data_preprocessing_in_python\\Part3'

# Data reduction

### distinction of Data redundancy and Data reduction.

Data redundancy is about having the same information presented under more than one attribute.

Data reduction is about reducing the size of data due to one of the following three reasons

- High-Dimensional Visualizations : 사람들은 3~5 차원 이상의 그래프를 이해하는데 어려움을 느낀다.

- Computational Cost : 불필요하게 많은 계산을 필요로 한다.
- Curse of Dimensionality : variable이 많다고 정확도가 높아지는 건 아니다.

### the objectives of data reduction

1. data reduction seeks to obtain a reduced representation of the dataset that is much smaller in volume.
2. it tries to closely maintain the integrity of the original data, which means making sure that data reduction will not lead to including bias and critical information being lost in the data.

### Types of data reduction

- Numerosity data reduction : It performs data reduction by reducing the number of data objects or rows in a dataset.

  - Random Sampling
  - Strafied Sampling
  - Random over/under sampling

- dimenionality data reduction : It performs data reduction by rducing the number of dimensions or attributes in a dataset.
  - Linear regression
  - Decision Tree
  - Random Forest
  - Brute-force Computational Dimension reduction
  - principal component analysis
  - Functional data analysis

### Numerosity data reduction

```python
customer_df = pd.read_csv('data/ch13/Customer Churn.csv')
```

```python
y = customer_df['Churn']
xs = customer_df.drop(columns='Churn')
param_grid = {'criterion' : ['gini','entropy'], 'max_depth' : [10,20,30,40,50,60], 'min_samples_split' : [10,20,30,40,50], 'min_impurity_decrease' : [0,0.001,0.005,0.01,0.05,0.1]}
gridsearch = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3, scoring='recall', verbose=1)
gridsearch.fit(xs,y)
print('Best Score : ', gridsearch.best_score_)
print('Best parameters : ', gridsearch.best_params_)
```

    Fitting 3 folds for each of 360 candidates, totalling 1080 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    Best Score :  0.7353535353535353
    Best parameters :  {'criterion': 'entropy', 'max_depth': 10, 'min_impurity_decrease': 0.005, 'min_samples_split': 40}


    [Parallel(n_jobs=1)]: Done 1080 out of 1080 | elapsed:   15.0s finished

### Random sampling

```python
customer_df_rs = customer_df.sample(1000, random_state=1)
y = customer_df_rs['Churn']
xs = customer_df_rs.drop(columns='Churn')
param_grid = {'criterion' : ['gini','entropy'], 'max_depth' : [10,20,30,40,50,60], 'min_samples_split' : [10,20,30,40,50], 'min_impurity_decrease' : [0,0.001,0.005,0.01,0.05,0.1]}
gridsearch = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3, scoring='recall', verbose=1)
gridsearch.fit(xs,y)
print('Best Score : ', gridsearch.best_score_)
print('Best parameters : ', gridsearch.best_params_)
```

    Fitting 3 folds for each of 360 candidates, totalling 1080 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    Best Score :  0.7430555555555555
    Best parameters :  {'criterion': 'entropy', 'max_depth': 10, 'min_impurity_decrease': 0.005, 'min_samples_split': 10}


    [Parallel(n_jobs=1)]: Done 1080 out of 1080 | elapsed:   11.7s finished

### Strified Sampling

```python
n, s = 1000, len(customer_df)
r = n/s
sample_df = customer_df.groupby('Churn', group_keys=False).apply(lambda sdf : sdf.sample(round(len(sdf)*r)))

print(sample_df.Churn.value_counts(normalize=True))
```

    0    0.843
    1    0.157
    Name: Churn, dtype: float64

### Random over/undersampling

random over/undersampling은 표본을 5:5 비율로 추출한다.

over/undersampling을 사용하기 위한 조건 두가지

1.  the dependent attribute is binary, meaning that it only has two class labels.
2.  there are significantly more of one class label than the other.

```python

sample_df = customer_df.groupby('Churn', group_keys=False).apply(lambda sdf : sdf.sample(round(250)))

print(sample_df.Churn.value_counts(normalize=True))
```

    1    0.5
    0    0.5
    Name: Churn, dtype: float64

```python
amzn_df = pd.read_csv('data/ch13/amznStock.csv', index_col='t')
```

```python
amzn_df.columns = ['pd_changeP', 'pw_changeP', 'dow_pd_changeP','dow_pw_changeP', 'nasdaq_pd_changeP', 'nasdaq_pw_changeP', 'changeP']
```

```python
import statsmodels.api as sm
xs = amzn_df.drop(columns=['changeP'], index=['2021-01-12'])
xs = sm.add_constant(xs)
y = amzn_df.drop(index=['2021-01-12']).changeP
sm.OLS(y,xs).fit().summary()
```

<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>changeP</td>     <th>  R-squared:         </th> <td>   0.061</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.044</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3.678</td>
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 26 Mar 2022</td> <th>  Prob (F-statistic):</th>  <td>0.00149</td>
</tr>
<tr>
  <th>Time:</th>                 <td>17:50:22</td>     <th>  Log-Likelihood:    </th> <td> -750.72</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>   349</td>      <th>  AIC:               </th> <td>   1515.</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   342</td>      <th>  BIC:               </th> <td>   1542.</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     6</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
          <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>             <td>    0.2342</td> <td>    0.122</td> <td>    1.926</td> <td> 0.055</td> <td>   -0.005</td> <td>    0.473</td>
</tr>
<tr>
  <th>pd_changeP</th>        <td>   -0.0804</td> <td>    0.112</td> <td>   -0.719</td> <td> 0.473</td> <td>   -0.300</td> <td>    0.140</td>
</tr>
<tr>
  <th>pw_changeP</th>        <td>    0.0665</td> <td>    0.044</td> <td>    1.499</td> <td> 0.135</td> <td>   -0.021</td> <td>    0.154</td>
</tr>
<tr>
  <th>dow_pd_changeP</th>    <td>   -0.2888</td> <td>    0.151</td> <td>   -1.914</td> <td> 0.056</td> <td>   -0.586</td> <td>    0.008</td>
</tr>
<tr>
  <th>dow_pw_changeP</th>    <td>    0.0866</td> <td>    0.066</td> <td>    1.316</td> <td> 0.189</td> <td>   -0.043</td> <td>    0.216</td>
</tr>
<tr>
  <th>nasdaq_pd_changeP</th> <td>    0.0919</td> <td>    0.210</td> <td>    0.438</td> <td> 0.661</td> <td>   -0.321</td> <td>    0.505</td>
</tr>
<tr>
  <th>nasdaq_pw_changeP</th> <td>   -0.1403</td> <td>    0.098</td> <td>   -1.433</td> <td> 0.153</td> <td>   -0.333</td> <td>    0.052</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>25.863</td> <th>  Durbin-Watson:     </th> <td>   1.936</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td>  97.802</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.036</td> <th>  Prob(JB):          </th> <td>5.79e-22</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.592</td> <th>  Cond. No.          </th> <td>    17.6</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

### Attribute 정리하는 방법은 다른 책에서 배우자 너무 부실하다.
