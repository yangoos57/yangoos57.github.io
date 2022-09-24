---
title: "6.5 Hyperparameter tuning with ensemble methods "
category: "MachineLearning"
date: "2022-04-18"
thumbnail: "./images/scikit-learn-logo.png"
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Random forest

모델 자체가 복잡하다 보니 같은 데이터를 처리하더라도 간단한 모델에 보다 몇십배 많은 시간이 걸린다.

효과적인 모델을 구축하기 위해서 보다 높은 hyperparameter을 설정할 수 있지만 computing cost와 generalization performance 둘다 만족할 수 있는 값을 구해야한다.

- n_estimators => main parameter
- max_depth => Symmetric tree 개수를 결정함
- max_leaf_nodes => node 당 최대 leaf 개수

> ❗Be aware that with random forest, trees are generally deep since we are seeking to overfit each tree on each bootstrap sample because this will be mitigated by combining them altogether.

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0)
```

```python
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

param_distributions = {
    "n_estimators": [1, 2, 5, 10, 20, 50, 100, 200, 500],
    "max_leaf_nodes": [2, 5, 10, 20, 50, 100],
}
search_cv = RandomizedSearchCV(
    RandomForestRegressor(n_jobs=2), param_distributions=param_distributions,
    scoring="neg_mean_absolute_error", n_iter=10, random_state=0, n_jobs=2,
)
search_cv.fit(data_train, target_train)

columns = [f"param_{name}" for name in param_distributions.keys()]
columns += ["mean_test_error", "std_test_error"]
cv_results = pd.DataFrame(search_cv.cv_results_)
cv_results["mean_test_error"] = -cv_results["mean_test_score"]
cv_results["std_test_error"] = cv_results["std_test_score"]
cv_results[columns].sort_values(by="mean_test_error")
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
      <th>param_n_estimators</th>
      <th>param_max_leaf_nodes</th>
      <th>mean_test_error</th>
      <th>std_test_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>500</td>
      <td>100</td>
      <td>40.856181</td>
      <td>0.693953</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>100</td>
      <td>41.285779</td>
      <td>0.840210</td>
    </tr>
    <tr>
      <th>7</th>
      <td>100</td>
      <td>50</td>
      <td>44.084379</td>
      <td>0.794622</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>100</td>
      <td>47.136405</td>
      <td>0.956875</td>
    </tr>
    <tr>
      <th>6</th>
      <td>50</td>
      <td>20</td>
      <td>49.334952</td>
      <td>0.748772</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100</td>
      <td>20</td>
      <td>49.511881</td>
      <td>0.867570</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>20</td>
      <td>49.955527</td>
      <td>0.892436</td>
    </tr>
    <tr>
      <th>3</th>
      <td>500</td>
      <td>10</td>
      <td>54.604095</td>
      <td>0.808515</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>61.440435</td>
      <td>0.975260</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>2</td>
      <td>73.063948</td>
      <td>0.693810</td>
    </tr>
  </tbody>
</table>
</div>

RandomizedSearchCV : 최적의 model 하나만을 추출한다.

```python
error = -search_cv.score(data_test, target_test)
print(f"On average, our random forest regressor makes an error of {error:.2f} k$")
```

    On average, our random forest regressor makes an error of 41.91 k$

### Gradient-boosting decision trees

- n_estimators : tree 개수

- max_depth : tree 변수

  > With this consideration in mind, the deeper the trees, the faster the residuals will be corrected and less learners are required. Therefore, n_estimators should be increased if max_depth is lower.

- max_leaf_nodes : tree 변수
- Learning_rate : online learning과 관련있는걸로 아는데 의미는 내가 아는 것과 다른가 봄
  > we would like the tree to try to correct all possible errors or only a fraction of them. A small learning-rate value would only correct the residuals of very few samples. If a large learning-rate is set (e.g., 1), we would fit the residuals of all samples. So, with a very low learning-rate, we will need more estimators to correct the overall error.

```python
from sklearn.ensemble import HistGradientBoostingRegressor

hist_gbdt = HistGradientBoostingRegressor(max_iter=1000, early_stopping=True, random_state=0)
```

```python
from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [3, 8],
    'max_leaf_nodes': [15, 31],
    'learning_rate': [0.1, 1],
}

grid_cv = GridSearchCV(estimator=hist_gbdt, param_grid=params,n_jobs=2)
```

```python
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, shuffle = True, random_state= 0)
cv_results = cross_validate(grid_cv,data,target, n_jobs=2, cv=cv, return_estimator=True)
```

```python
cv_results['test_score'].std()
```

    0.031483767082097436

```python
print(f"{cv_results['test_score'].mean():.3f} {cv_results['test_score'].std():.3f}")
```

    0.839 0.006

```python
# solution
for estimator in cv_results["estimator"]:
    print(estimator.best_params_)
    print(f"# trees: {estimator.best_estimator_.n_iter_}")
```

    {'learning_rate': 0.1, 'max_depth': 3, 'max_leaf_nodes': 15}
    # trees: 528
    {'learning_rate': 0.1, 'max_depth': 8, 'max_leaf_nodes': 15}
    # trees: 447
    {'learning_rate': 0.1, 'max_depth': 3, 'max_leaf_nodes': 15}
    # trees: 576
    {'learning_rate': 0.1, 'max_depth': 8, 'max_leaf_nodes': 15}
    # trees: 290
    {'learning_rate': 0.1, 'max_depth': 8, 'max_leaf_nodes': 15}
    # trees: 414
