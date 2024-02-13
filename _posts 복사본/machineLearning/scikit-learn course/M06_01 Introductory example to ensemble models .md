---
title: "6.1 Introductory example to ensemble models "
category: "MachineLearning"
date: "2022-04-14"
thumbnail: "./images/scikit-learn-logo.png"
---

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeRegressor


data, target = fetch_california_housing(as_frame=True, return_X_y=True)
target *= 100  # rescale the target in k$

tree = DecisionTreeRegressor(random_state=0)
cv_results = cross_validate(tree, data, target, n_jobs=2)
scores = cv_results["test_score"]

print(f"R2 score obtained by cross-validation: "
      f"{scores.mean():.3f} +/- {scores.std():.3f}")
```

    R2 score obtained by cross-validation: 0.354 +/- 0.087

```python
%%time
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

param_grid = {
    "max_depth": [5, 8, None],
    "min_samples_split": [2, 10, 30, 50],
    "min_samples_leaf": [0.01, 0.05, 0.1, 1]}
cv = 3

tree = GridSearchCV(DecisionTreeRegressor(random_state=0),
                    param_grid=param_grid, cv=cv, n_jobs=2)
cv_results = cross_validate(tree, data, target, n_jobs=2,
                            return_estimator=True)
scores = cv_results["test_score"]

print(f"R2 score obtained by cross-validation: "
      f"{scores.mean():.3f} +/- {scores.std():.3f}")
```

    R2 score obtained by cross-validation: 0.523 +/- 0.107
    Wall time: 17.7 s

### Bagging Regressor

- this method will use a base regressor (i.e. decision tree regressors) and will train several of them on a slightly modified version of the training set.

- Then, the predictions of all these base regressors will be combined by averaging.

```python
%%time
from sklearn.ensemble import BaggingRegressor

base_estimator = DecisionTreeRegressor(random_state=0)
bagging_regressor = BaggingRegressor(
    base_estimator=base_estimator, n_estimators=20, random_state=0)

cv_results = cross_validate(bagging_regressor, data, target, n_jobs=2)
scores = cv_results["test_score"]

print(f"R2 score obtained by cross-validation: "
      f"{scores.mean():.3f} +/- {scores.std():.3f}")
```

    R2 score obtained by cross-validation: 0.642 +/- 0.083
    Wall time: 7.6 s

- We will use 20 decision trees and check the fitting time as well as the generalization performance on the left-out testing data.

- It is important to note that we are not going to tune any parameter of the decision tree.
