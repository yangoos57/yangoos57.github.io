---
title: "2.1 Manual tuning"
category: "MachineLearning"
date: "2022-04-03"
thumbnail: "./images/scikit-learn-logo.png"
---

```python
import pandas as pd
import numpy as np
import os
os.getcwd()

```

    'd:\\git_local_repository\\yangoos57\\ML\\[inria] scikit-learn course'

### Set and get Hyperparameters in scikit-learn

We recall that hyperparameters refer to the parameter that will control the learning process

```python
adult_df = pd.read_csv('data/phpMawTba.csv').drop(columns='education-num')

target = adult_df['class']
data = adult_df[['age','capital-gain','capital-loss','hours-per-week']]
```

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = Pipeline(steps=[('preprocessor',StandardScaler()),('classifier',LogisticRegression())])

```

```python
from sklearn.model_selection import cross_val_score

cv_results = cross_val_score(model,data,target)
cv_results
```

    array([0.79557785, 0.80049135, 0.79965192, 0.79873055, 0.80436118])

- Linear model에서 C라는 parameter은 모델을 얼마나 regularization하는지를 결정한다.

regularization ?

- Pipeline에서 parameter을 수정하려면 <model_name>\_\_<parameter_name>으로 해야함

```python
model.set_params(classifier__C = 1e-3)
cv_results = cross_val_score(model,data,target)
cv_results.mean()
```

    0.7873552003785396

```python
for param in model.get_params() :
    print(param)
```

    memory
    steps
    verbose
    preprocessor
    classifier
    preprocessor__copy
    preprocessor__with_mean
    preprocessor__with_std
    classifier__C
    classifier__class_weight
    classifier__dual
    classifier__fit_intercept
    classifier__intercept_scaling
    classifier__l1_ratio
    classifier__max_iter
    classifier__multi_class
    classifier__n_jobs
    classifier__penalty
    classifier__random_state
    classifier__solver
    classifier__tol
    classifier__verbose
    classifier__warm_start

```python
model.get_params()['classifier__C']
```

    0.001

```python
for c in [1e-3,1e-2,1e-1,1,10] :
    model.set_params(classifier__C=c)
    cv_results = cross_val_score(model,data,target)
    print(cv_results.mean(), '|', cv_results.std())
```

    0.7873552003785396 | 0.0018873549930111077
    0.799332617870851 | 0.003165928979859665
    0.7995987637941779 | 0.002778046832848365
    0.7997625702457313 | 0.0028378701400111547
    0.7997830452662062 | 0.0028046734661818177

최적의 모델을 찾기 위해서는 하나의 테스트셋을 여러 모델에 적용한 뒤 그 결과값인 test_score을 비교하면 된다.

최적의 모델을 선정하고 난 뒤에는 새로운 테스트 셋으로 model_score을 추출해야한다. 왜냐하면 test_score은 모델을 선택할때 이미 사용했기 때문이다.

이 말의 50%정도만 이해한듯.test_score이 independent가 더이상 아니라는데, 새로운 test를 통해 나온 accuracy라는 사실은 변함없지 않나.

그리고 test_data라는 말이 개별적으로 train한 뒤 test_set으로 다시 scoring을 찾았다는 말 아닌가? 그런데 굳이 모델을 선별한 뒤 또 한번 새로운 test_set을 구해서 점수를 메기는걸까?

뒷장에서 설명이 나온다.

### Exercise

```python
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(data, target, train_size=0.2, random_state=42)
```

### columntransformer : 데이터 가공 없이 원하는 type을 추출하여 원하는 preprocessor를 적용시킬 수 있도록 돕는 절차임.

쉽게 생각해 데이터를 가공할 컨베이어 벨트를 미리 만들어 놓는다고 보면 될 듯. 원하는 데이터 유형(categorical 또는 numerical)을 택한 뒤 해당 columns에 preprocessing을 수행한다.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_selector as selector
categorical_preprocessor = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)

preprocessor = ColumnTransformer([('categorical', categorical_preprocessor,selector(dtype_include='object'))], remainder='passthrough', sparse_threshold=0)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

model = Pipeline([('preprocessor', preprocessor),('classifier',HistGradientBoostingClassifier(random_state=42))])

```

```python
from sklearn.model_selection import cross_val_score

learning_rate = [0.01,0.1,1,10]
max_leaf_nodes = [3, 10, 30]

best_score = 0
best_params ={}

for lr in learning_rate :
    for mln in max_leaf_nodes :
        print(lr,"||",mln)
        model.set_params(classifier__learning_rate = lr, classifier__max_leaf_nodes = mln)
        scores = cross_val_score(model, data_train, target_train, cv=2)
        mean_score =scores.mean()
        print(f'score : {mean_score : .3f}')
        if mean_score > best_score :
            best_score = mean_score
            best_params = {'learning-rate' : lr, 'max_leaf_nodes': mln}
            print(f'best_score : {best_score}')
print(f'final_best_score : {best_score}')
print(f'max_leaf_nodes : {best_params}')
```

    0.01 || 3
    score :  0.794
    best_score : 0.7937141687141687
    0.01 || 10
    score :  0.810
    best_score : 0.8100941850941851
    0.01 || 30
    score :  0.810
    best_score : 0.8100941850941852
    0.1 || 3
    score :  0.812
    best_score : 0.8123464373464373
    0.1 || 10
    score :  0.820
    best_score : 0.8201269451269451
    0.1 || 30
    score :  0.815
    1 || 3
    score :  0.816
    1 || 10
    score :  0.806
    1 || 30
    score :  0.800
    10 || 3
    score :  0.545
    10 || 10
    score :  0.469
    10 || 30
    score :  0.580
    final_best_score : 0.8201269451269451
    max_leaf_nodes : {'learning-rate': 0.1, 'max_leaf_nodes': 10}

```python
best_lr = best_params['learning-rate']
best_mln = best_params['max_leaf_nodes']

model.set_params(classifier__learning_rate = best_lr, classifier__max_leaf_nodes = best_mln)

model.fit(data_train,target_train)

test_score = model.score(data_test, target_test)

test_score
```

    0.8286584429543943

```python

```
