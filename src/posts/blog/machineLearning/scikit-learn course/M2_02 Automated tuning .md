---
title: "M2_02 Automated tuning "
category: "MachineLearning"
date: "2021-05-01"
thumbnail: "./images/scikit-learn-logo.png"
---

```python
import pandas as pd
import numpy as np
import os
os.getcwd()

```

    'd:\\git_local_repository\\yangoos57\\ML\\[inria] scikit-learn course'

```python
from sklearn import set_config
set_config(display='diagram')
```

```python
adult_df = pd.read_csv('data/phpMawTba.csv').drop(columns='education-num')

target = adult_df['class']
```

```python
data = adult_df.drop(columns=['class'])
```

```python
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=42)
```

```python
from sklearn.compose import make_column_selector as selector
categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
# categorical_columns_2 = data.select_dtypes('object').columns.to_list()

# print(categorical_columns)
# print(categorical_columns_2)
```

```python
from sklearn.preprocessing import OrdinalEncoder

categorical_preprocessor = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value= -1)
```

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([('categorical', categorical_preprocessor,categorical_columns)], remainder='passthrough', sparse_threshold=0)
```

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

model = Pipeline(steps=[('preprocessor',preprocessor), ('classifier', HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=4))])
```

```python
model
```

<style>#sk-1578d6d9-e579-4385-95f7-388b809f5864 {color: black;background-color: white;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 pre{padding: 0;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-toggleable {background-color: white;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-estimator:hover {background-color: #d4ebff;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-item {z-index: 1;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-parallel-item:only-child::after {width: 0;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-1578d6d9-e579-4385-95f7-388b809f5864 div.sk-text-repr-fallback {display: none;}</style><div id="sk-1578d6d9-e579-4385-95f7-388b809f5864" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,

                 ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,
                                   transformers=[(&#x27;categorical&#x27;,
                                                  OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,
                                                                 unknown_value=-1),
                                                  [&#x27;workclass&#x27;, &#x27;education&#x27;,
                                                   &#x27;marital-status&#x27;,
                                                   &#x27;occupation&#x27;, &#x27;relationship&#x27;,
                                                   &#x27;race&#x27;, &#x27;sex&#x27;,
                                                   &#x27;native-country&#x27;])])),
                (&#x27;classifier&#x27;,
                 HistGradientBoostingClassifier(max_leaf_nodes=4,
                                                random_state=42))])</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="ec6b7816-2f69-48f5-8244-f5254be40777" type="checkbox" ><label for="ec6b7816-2f69-48f5-8244-f5254be40777" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,
                                   transformers=[(&#x27;categorical&#x27;,
                                                  OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,
                                                                 unknown_value=-1),
                                                  [&#x27;workclass&#x27;, &#x27;education&#x27;,
                                                   &#x27;marital-status&#x27;,
                                                   &#x27;occupation&#x27;, &#x27;relationship&#x27;,
                                                   &#x27;race&#x27;, &#x27;sex&#x27;,
                                                   &#x27;native-country&#x27;])])),
                (&#x27;classifier&#x27;,
                 HistGradientBoostingClassifier(max_leaf_nodes=4,
                                                random_state=42))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="bae2abb7-5613-4e30-8a32-3e0035bee2d9" type="checkbox" ><label for="bae2abb7-5613-4e30-8a32-3e0035bee2d9" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,
                  transformers=[(&#x27;categorical&#x27;,
                                 OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,
                                                unknown_value=-1),
                                 [&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;,
                                  &#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;,
                                  &#x27;native-country&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="d077f96a-2214-430b-a29f-4888a70e3bc6" type="checkbox" ><label for="d077f96a-2214-430b-a29f-4888a70e3bc6" class="sk-toggleable__label sk-toggleable__label-arrow">categorical</label><div class="sk-toggleable__content"><pre>[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;, &#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;, &#x27;native-country&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="ab198658-d577-485d-a10f-cd588477532c" type="checkbox" ><label for="ab198658-d577-485d-a10f-cd588477532c" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;, unknown_value=-1)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="1ac87142-aa27-4353-940f-8a6e0582d1f5" type="checkbox" ><label for="1ac87142-aa27-4353-940f-8a6e0582d1f5" class="sk-toggleable__label sk-toggleable__label-arrow">remainder</label><div class="sk-toggleable__content"><pre></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="4b7b892d-d24b-4cb9-b207-59e0f1425c54" type="checkbox" ><label for="4b7b892d-d24b-4cb9-b207-59e0f1425c54" class="sk-toggleable__label sk-toggleable__label-arrow">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c2c88240-7bea-4121-8a8f-5f5dfb0323e2" type="checkbox" ><label for="c2c88240-7bea-4121-8a8f-5f5dfb0323e2" class="sk-toggleable__label sk-toggleable__label-arrow">HistGradientBoostingClassifier</label><div class="sk-toggleable__content"><pre>HistGradientBoostingClassifier(max_leaf_nodes=4, random_state=42)</pre></div></div></div></div></div></div></div>

### Tuning using a grid-search

cv: cross-validation, the number of cv folds for each combination of parameters

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__learning_rate' : (0.01,0.1,1,10),
    'classifier__max_leaf_nodes' : (3,10,30)
}

model_grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=2, cv=2)

model_grid_search.fit(data_train,target_train)
```

<style>#sk-6a844ee9-43de-4df7-8931-697ade029203 {color: black;background-color: white;}#sk-6a844ee9-43de-4df7-8931-697ade029203 pre{padding: 0;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-toggleable {background-color: white;}#sk-6a844ee9-43de-4df7-8931-697ade029203 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-6a844ee9-43de-4df7-8931-697ade029203 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-6a844ee9-43de-4df7-8931-697ade029203 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-6a844ee9-43de-4df7-8931-697ade029203 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-6a844ee9-43de-4df7-8931-697ade029203 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-6a844ee9-43de-4df7-8931-697ade029203 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-estimator:hover {background-color: #d4ebff;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-item {z-index: 1;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-parallel-item:only-child::after {width: 0;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-6a844ee9-43de-4df7-8931-697ade029203 div.sk-text-repr-fallback {display: none;}</style><div id="sk-6a844ee9-43de-4df7-8931-697ade029203" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=2,

             estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,
                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                          sparse_threshold=0,
                                                          transformers=[(&#x27;categorical&#x27;,
                                                                         OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,
                                                                                        unknown_value=-1),
                                                                         [&#x27;workclass&#x27;,
                                                                          &#x27;education&#x27;,
                                                                          &#x27;marital-status&#x27;,
                                                                          &#x27;occupation&#x27;,
                                                                          &#x27;relationship&#x27;,
                                                                          &#x27;race&#x27;,
                                                                          &#x27;sex&#x27;,
                                                                          &#x27;native-country&#x27;])])),
                                       (&#x27;classifier&#x27;,
                                        HistGradientBoostingClassifier(max_leaf_nodes=4,
                                                                       random_state=42))]),
             n_jobs=2,
             param_grid={&#x27;classifier__learning_rate&#x27;: (0.01, 0.1, 1, 10),
                         &#x27;classifier__max_leaf_nodes&#x27;: (3, 10, 30)})</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="34808d1e-1880-4330-9d35-a5bc1d25af44" type="checkbox" ><label for="34808d1e-1880-4330-9d35-a5bc1d25af44" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=2,
             estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,
                                        ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                          sparse_threshold=0,
                                                          transformers=[(&#x27;categorical&#x27;,
                                                                         OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,
                                                                                        unknown_value=-1),
                                                                         [&#x27;workclass&#x27;,
                                                                          &#x27;education&#x27;,
                                                                          &#x27;marital-status&#x27;,
                                                                          &#x27;occupation&#x27;,
                                                                          &#x27;relationship&#x27;,
                                                                          &#x27;race&#x27;,
                                                                          &#x27;sex&#x27;,
                                                                          &#x27;native-country&#x27;])])),
                                       (&#x27;classifier&#x27;,
                                        HistGradientBoostingClassifier(max_leaf_nodes=4,
                                                                       random_state=42))]),
             n_jobs=2,
             param_grid={&#x27;classifier__learning_rate&#x27;: (0.01, 0.1, 1, 10),
                         &#x27;classifier__max_leaf_nodes&#x27;: (3, 10, 30)})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="8d490b36-3b6d-4e8e-8191-4157e5f13c37" type="checkbox" ><label for="8d490b36-3b6d-4e8e-8191-4157e5f13c37" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,
                  transformers=[(&#x27;categorical&#x27;,
                                 OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,
                                                unknown_value=-1),
                                 [&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;,
                                  &#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;,
                                  &#x27;native-country&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c841b2f5-809c-4bb2-9fbd-117c6ba07135" type="checkbox" ><label for="c841b2f5-809c-4bb2-9fbd-117c6ba07135" class="sk-toggleable__label sk-toggleable__label-arrow">categorical</label><div class="sk-toggleable__content"><pre>[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;, &#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;, &#x27;native-country&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="39e8645a-f201-403b-a682-54df85d2a7a3" type="checkbox" ><label for="39e8645a-f201-403b-a682-54df85d2a7a3" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;, unknown_value=-1)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c413a09b-1df4-45af-a077-680688e1da7f" type="checkbox" ><label for="c413a09b-1df4-45af-a077-680688e1da7f" class="sk-toggleable__label sk-toggleable__label-arrow">remainder</label><div class="sk-toggleable__content"><pre></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="f4f271dc-d21f-49f3-b407-ba9e625124d2" type="checkbox" ><label for="f4f271dc-d21f-49f3-b407-ba9e625124d2" class="sk-toggleable__label sk-toggleable__label-arrow">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="08067d5a-3c77-4bbe-ba68-2a34c062e9b2" type="checkbox" ><label for="08067d5a-3c77-4bbe-ba68-2a34c062e9b2" class="sk-toggleable__label sk-toggleable__label-arrow">HistGradientBoostingClassifier</label><div class="sk-toggleable__content"><pre>HistGradientBoostingClassifier(max_leaf_nodes=4, random_state=42)</pre></div></div></div></div></div></div></div></div></div></div></div></div>

```python
accuracy = model_grid_search.score(data_test,target_test)
print(f'accuracy : {accuracy : .3f}')
```

    accuracy :  0.880

### GridsearchCV : param_grid

모델에 있는 parameter를 조절하는 kwarg임. 위에서는 4\*3이므로 12가지. hyperparam이 많아질수록 연산 시간은 오래걸린다.

```python
print(f'best_param : {model_grid_search.best_params_}')
```

    best_param : {'classifier__learning_rate': 0.1, 'classifier__max_leaf_nodes': 30}

```python
cv_results = pd.DataFrame(model_grid_search.cv_results_).sort_values('mean_test_score',ascending=False)
cv_results.head()
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
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_classifier__learning_rate</th>
      <th>param_classifier__max_leaf_nodes</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>1.121222</td>
      <td>0.051359</td>
      <td>0.182035</td>
      <td>0.001357</td>
      <td>0.1</td>
      <td>30</td>
      <td>{'classifier__learning_rate': 0.1, 'classifier...</td>
      <td>0.867766</td>
      <td>0.867649</td>
      <td>0.867708</td>
      <td>0.000058</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.786589</td>
      <td>0.010644</td>
      <td>0.186672</td>
      <td>0.009130</td>
      <td>0.1</td>
      <td>10</td>
      <td>{'classifier__learning_rate': 0.1, 'classifier...</td>
      <td>0.866729</td>
      <td>0.866557</td>
      <td>0.866643</td>
      <td>0.000086</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.221054</td>
      <td>0.003271</td>
      <td>0.111194</td>
      <td>0.006268</td>
      <td>1</td>
      <td>3</td>
      <td>{'classifier__learning_rate': 1, 'classifier__...</td>
      <td>0.860559</td>
      <td>0.861261</td>
      <td>0.860910</td>
      <td>0.000351</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.205914</td>
      <td>0.003294</td>
      <td>0.108691</td>
      <td>0.011758</td>
      <td>1</td>
      <td>10</td>
      <td>{'classifier__learning_rate': 1, 'classifier__...</td>
      <td>0.857993</td>
      <td>0.861862</td>
      <td>0.859927</td>
      <td>0.001934</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.372542</td>
      <td>0.005944</td>
      <td>0.144263</td>
      <td>0.006093</td>
      <td>0.1</td>
      <td>3</td>
      <td>{'classifier__learning_rate': 0.1, 'classifier...</td>
      <td>0.852752</td>
      <td>0.854272</td>
      <td>0.853512</td>
      <td>0.000760</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>

```python
column_results = [f'param_{name}' for name in param_grid.keys()]
column_results += ['mean_test_score','std_test_score','rank_test_score']
cv_results = cv_results[column_results]
```

```python
cv_results.columns = ['learning_rate','max_leaf_nodes','mean_test_score','std_test_score','rank_test_score']
```

```python
a = cv_results.pivot_table(index=['learning_rate'],columns=['max_leaf_nodes'], values='mean_test_score')

a
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
      <th>max_leaf_nodes</th>
      <th>3</th>
      <th>10</th>
      <th>30</th>
    </tr>
    <tr>
      <th>learning_rate</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.01</th>
      <td>0.797166</td>
      <td>0.817832</td>
      <td>0.845541</td>
    </tr>
    <tr>
      <th>0.10</th>
      <td>0.853512</td>
      <td>0.866643</td>
      <td>0.867708</td>
    </tr>
    <tr>
      <th>1.00</th>
      <td>0.860910</td>
      <td>0.859927</td>
      <td>0.851547</td>
    </tr>
    <tr>
      <th>10.00</th>
      <td>0.283476</td>
      <td>0.618080</td>
      <td>0.351642</td>
    </tr>
  </tbody>
</table>
</div>

```python
import seaborn as sns

ax = sns.heatmap(a, annot=True, cmap='YlGnBu', vmin=0.7, vmax=0.9)
ax.invert_yaxis()
```

![png](output_20_0.png)

1. learning_rate가 너무 높으니 결과가 좋지 않음.
2. 결과에는 learning_rate가 max_leaf_nodes보다 더 큰 영향을 미침
3. hyperparameter들은 모두 중요함.
4. 두 개 이상의 parameter를 사용하는 것은 연산 영역에 있어서 큰 비용을 초래함.
5. grid-search가 매번 optimal solution을 찾지는 않음.

### Tuning using a randomized-search

- exploring a large number of values for different parameters will be quickly untractable.
- RandomizedSearchCV : stochastic search. It is typically beneficial compared to grid search to optimize 3 or more hyperparameters.
  ![photo](data/9.png)

```python
from scipy.stats import loguniform



class loguniform_int :
    ### integer vlaued version of the log-uniform distribution
    def __init__(self,a,b) :
        self._distribution = loguniform(a,b)
    def rvs(self, *args, **kwargs) :
        ### random variable sample
        return self._distribution.rvs(*args, **kwargs).astype(int)
```

```python
%%time
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    'classifier__l2_regularization' : loguniform(1e-6,1e3),
    'classifier__learning_rate' : loguniform(0.001,10),
    'classifier__max_leaf_nodes' : loguniform_int(2, 256),
    'classifier__min_samples_leaf' : loguniform_int(1,100),
    'classifier__max_bins' : loguniform_int(2,255),
}

model_random_search = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=20, cv=5, n_jobs=-1)

model_random_search.fit(data_train, target_train)
```

    Wall time: 48 s

<style>#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 {color: black;background-color: white;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 pre{padding: 0;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-toggleable {background-color: white;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-estimator:hover {background-color: #d4ebff;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-item {z-index: 1;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-parallel-item:only-child::after {width: 0;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022 div.sk-text-repr-fallback {display: none;}</style><div id="sk-6cfbc36f-c1a4-4dc3-8cba-72ea67f8d022" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomizedSearchCV(cv=5,

                   estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,
                                              ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                                sparse_threshold=0,
                                                                transformers=[(&#x27;categorical&#x27;,
                                                                               OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,
                                                                                              unknown_value=-1),
                                                                               [&#x27;workclass&#x27;,
                                                                                &#x27;education&#x27;,
                                                                                &#x27;marital-status&#x27;,
                                                                                &#x27;occupation&#x27;,
                                                                                &#x27;relationship&#x27;,
                                                                                &#x27;race&#x27;,
                                                                                &#x27;sex&#x27;,
                                                                                &#x27;native-country&#x27;])])),
                                             (&#x27;classifier&#x27;,
                                              HistGra...
                   param_distributions={&#x27;classifier__l2_regularization&#x27;: &lt;scipy.stats._distn_infrastructure.rv_frozen object at 0x000002A2C4E242E0&gt;,
                                        &#x27;classifier__learning_rate&#x27;: &lt;scipy.stats._distn_infrastructure.rv_frozen object at 0x000002A2C4DFFE80&gt;,
                                        &#x27;classifier__max_bins&#x27;: &lt;__main__.loguniform_int object at 0x000002A2C4E24DC0&gt;,
                                        &#x27;classifier__max_leaf_nodes&#x27;: &lt;__main__.loguniform_int object at 0x000002A2C4E248E0&gt;,
                                        &#x27;classifier__min_samples_leaf&#x27;: &lt;__main__.loguniform_int object at 0x000002A2C4E24AC0&gt;})</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="22a048a2-219c-43d4-a0ee-17f13637f815" type="checkbox" ><label for="22a048a2-219c-43d4-a0ee-17f13637f815" class="sk-toggleable__label sk-toggleable__label-arrow">RandomizedSearchCV</label><div class="sk-toggleable__content"><pre>RandomizedSearchCV(cv=5,
                   estimator=Pipeline(steps=[(&#x27;preprocessor&#x27;,
                                              ColumnTransformer(remainder=&#x27;passthrough&#x27;,
                                                                sparse_threshold=0,
                                                                transformers=[(&#x27;categorical&#x27;,
                                                                               OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,
                                                                                              unknown_value=-1),
                                                                               [&#x27;workclass&#x27;,
                                                                                &#x27;education&#x27;,
                                                                                &#x27;marital-status&#x27;,
                                                                                &#x27;occupation&#x27;,
                                                                                &#x27;relationship&#x27;,
                                                                                &#x27;race&#x27;,
                                                                                &#x27;sex&#x27;,
                                                                                &#x27;native-country&#x27;])])),
                                             (&#x27;classifier&#x27;,
                                              HistGra...
                   param_distributions={&#x27;classifier__l2_regularization&#x27;: &lt;scipy.stats._distn_infrastructure.rv_frozen object at 0x000002A2C4E242E0&gt;,
                                        &#x27;classifier__learning_rate&#x27;: &lt;scipy.stats._distn_infrastructure.rv_frozen object at 0x000002A2C4DFFE80&gt;,
                                        &#x27;classifier__max_bins&#x27;: &lt;__main__.loguniform_int object at 0x000002A2C4E24DC0&gt;,
                                        &#x27;classifier__max_leaf_nodes&#x27;: &lt;__main__.loguniform_int object at 0x000002A2C4E248E0&gt;,
                                        &#x27;classifier__min_samples_leaf&#x27;: &lt;__main__.loguniform_int object at 0x000002A2C4E24AC0&gt;})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="4b92c708-dd2d-4dda-b784-fd56e35f3c98" type="checkbox" ><label for="4b92c708-dd2d-4dda-b784-fd56e35f3c98" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,
                  transformers=[(&#x27;categorical&#x27;,
                                 OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,
                                                unknown_value=-1),
                                 [&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;,
                                  &#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;,
                                  &#x27;native-country&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="887979ec-61bd-4d0c-ab93-c17eb43a4758" type="checkbox" ><label for="887979ec-61bd-4d0c-ab93-c17eb43a4758" class="sk-toggleable__label sk-toggleable__label-arrow">categorical</label><div class="sk-toggleable__content"><pre>[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;, &#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;, &#x27;native-country&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="2dd6667f-2d43-4823-9c43-c2c131a35bd0" type="checkbox" ><label for="2dd6667f-2d43-4823-9c43-c2c131a35bd0" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;, unknown_value=-1)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="3d3ac793-14c0-4a66-b7d1-d38a0625560e" type="checkbox" ><label for="3d3ac793-14c0-4a66-b7d1-d38a0625560e" class="sk-toggleable__label sk-toggleable__label-arrow">remainder</label><div class="sk-toggleable__content"><pre></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="c5840d7a-a029-4292-8de3-79693b43f493" type="checkbox" ><label for="c5840d7a-a029-4292-8de3-79693b43f493" class="sk-toggleable__label sk-toggleable__label-arrow">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="48788c98-7dc3-4f35-a0e5-da397ed679c5" type="checkbox" ><label for="48788c98-7dc3-4f35-a0e5-da397ed679c5" class="sk-toggleable__label sk-toggleable__label-arrow">HistGradientBoostingClassifier</label><div class="sk-toggleable__content"><pre>HistGradientBoostingClassifier(max_leaf_nodes=4, random_state=42)</pre></div></div></div></div></div></div></div></div></div></div></div></div>

```python
accuracy =model_random_search.score(data_test, target_test)

print(f'accuracy : {accuracy: .2f}')
```

    accuracy :  0.88

```python
from pprint import pprint

pprint(model_random_search.best_params_)
```

    {'classifier__l2_regularization': 0.8869470640319301,
     'classifier__learning_rate': 0.13430838577142845,
     'classifier__max_bins': 159,
     'classifier__max_leaf_nodes': 55,
     'classifier__min_samples_leaf': 22}

```python
column_results = [
    f"param_{name}" for name in param_distributions.keys()
]
column_results += ['mean_test_score','std_test_score', 'rank_test_score']

cv_results = pd.DataFrame(model_random_search.cv_results_)

cv_results = cv_results[column_results].sort_values('mean_test_score', ascending=False)

# cv_results.to_csv('randomized_search_results.csv')

cv_results

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
      <th>param_classifier__l2_regularization</th>
      <th>param_classifier__learning_rate</th>
      <th>param_classifier__max_leaf_nodes</th>
      <th>param_classifier__min_samples_leaf</th>
      <th>param_classifier__max_bins</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>0.886947</td>
      <td>0.134308</td>
      <td>55</td>
      <td>22</td>
      <td>159</td>
      <td>0.868418</td>
      <td>0.001624</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.272267</td>
      <td>1.557292</td>
      <td>6</td>
      <td>65</td>
      <td>193</td>
      <td>0.863121</td>
      <td>0.003094</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>804.993447</td>
      <td>0.480762</td>
      <td>239</td>
      <td>8</td>
      <td>86</td>
      <td>0.858153</td>
      <td>0.001740</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.000082</td>
      <td>0.083902</td>
      <td>11</td>
      <td>3</td>
      <td>77</td>
      <td>0.857989</td>
      <td>0.002701</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.000062</td>
      <td>0.07973</td>
      <td>34</td>
      <td>1</td>
      <td>50</td>
      <td>0.855369</td>
      <td>0.003024</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.599662</td>
      <td>0.028235</td>
      <td>10</td>
      <td>28</td>
      <td>124</td>
      <td>0.854986</td>
      <td>0.002385</td>
      <td>6</td>
    </tr>
    <tr>
      <th>11</th>
      <td>21.5217</td>
      <td>0.132571</td>
      <td>14</td>
      <td>40</td>
      <td>7</td>
      <td>0.842347</td>
      <td>0.003795</td>
      <td>7</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2.607199</td>
      <td>0.058006</td>
      <td>33</td>
      <td>76</td>
      <td>9</td>
      <td>0.839207</td>
      <td>0.003414</td>
      <td>8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.000058</td>
      <td>0.051218</td>
      <td>7</td>
      <td>18</td>
      <td>5</td>
      <td>0.827605</td>
      <td>0.003070</td>
      <td>9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.000324</td>
      <td>0.039254</td>
      <td>14</td>
      <td>8</td>
      <td>4</td>
      <td>0.814037</td>
      <td>0.002486</td>
      <td>10</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.002164</td>
      <td>0.623764</td>
      <td>25</td>
      <td>17</td>
      <td>2</td>
      <td>0.802681</td>
      <td>0.001737</td>
      <td>11</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.011592</td>
      <td>0.099709</td>
      <td>5</td>
      <td>78</td>
      <td>2</td>
      <td>0.801780</td>
      <td>0.002669</td>
      <td>12</td>
    </tr>
    <tr>
      <th>9</th>
      <td>179.542793</td>
      <td>0.036169</td>
      <td>3</td>
      <td>25</td>
      <td>25</td>
      <td>0.801261</td>
      <td>0.002987</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.337553</td>
      <td>0.008691</td>
      <td>76</td>
      <td>4</td>
      <td>6</td>
      <td>0.799951</td>
      <td>0.001863</td>
      <td>14</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.003627</td>
      <td>0.009312</td>
      <td>7</td>
      <td>35</td>
      <td>10</td>
      <td>0.785018</td>
      <td>0.002276</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>185.383783</td>
      <td>0.011758</td>
      <td>41</td>
      <td>1</td>
      <td>15</td>
      <td>0.777975</td>
      <td>0.002354</td>
      <td>16</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.000001</td>
      <td>0.022697</td>
      <td>2</td>
      <td>24</td>
      <td>2</td>
      <td>0.772924</td>
      <td>0.001723</td>
      <td>17</td>
    </tr>
    <tr>
      <th>19</th>
      <td>56.524961</td>
      <td>2.930116</td>
      <td>8</td>
      <td>2</td>
      <td>218</td>
      <td>0.763806</td>
      <td>0.011950</td>
      <td>18</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.452465</td>
      <td>0.005245</td>
      <td>31</td>
      <td>5</td>
      <td>5</td>
      <td>0.758947</td>
      <td>0.000013</td>
      <td>19</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.025624</td>
      <td>0.002431</td>
      <td>7</td>
      <td>22</td>
      <td>92</td>
      <td>0.758947</td>
      <td>0.000013</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>

### 최적의 값을 도출하는 Parameter는 여러가지 일 수 있음.

RandomizedSearchCV는 GridSearchCV에 비해 더 많은 parameter를 사용할 수 있다.

GridSearchCV는 값을 지정하는데 반해 RandomizedSearchCV는 범위를 줘서 그 안에서 random으로 값을 선택하도록 한다.

잘은 모르겠지만 RandomizedSearchCV는 gridsearch에서 사용하는 regularity를 줄이기 때문에 오류가 발생할 수도 있다고 한다.

```python
def shorten_param(param_name) :
    if "__" in param_name :
        return param_name.rsplit("__",1)[1]
    return param_name

cv_results = cv_results.rename(shorten_param, axis=1)
cv_results
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
      <th>l2_regularization</th>
      <th>learning_rate</th>
      <th>max_leaf_nodes</th>
      <th>min_samples_leaf</th>
      <th>max_bins</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>0.886947</td>
      <td>0.134308</td>
      <td>55</td>
      <td>22</td>
      <td>159</td>
      <td>0.868418</td>
      <td>0.001624</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.272267</td>
      <td>1.557292</td>
      <td>6</td>
      <td>65</td>
      <td>193</td>
      <td>0.863121</td>
      <td>0.003094</td>
      <td>2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>804.993447</td>
      <td>0.480762</td>
      <td>239</td>
      <td>8</td>
      <td>86</td>
      <td>0.858153</td>
      <td>0.001740</td>
      <td>3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.000082</td>
      <td>0.083902</td>
      <td>11</td>
      <td>3</td>
      <td>77</td>
      <td>0.857989</td>
      <td>0.002701</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.000062</td>
      <td>0.07973</td>
      <td>34</td>
      <td>1</td>
      <td>50</td>
      <td>0.855369</td>
      <td>0.003024</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.599662</td>
      <td>0.028235</td>
      <td>10</td>
      <td>28</td>
      <td>124</td>
      <td>0.854986</td>
      <td>0.002385</td>
      <td>6</td>
    </tr>
    <tr>
      <th>11</th>
      <td>21.5217</td>
      <td>0.132571</td>
      <td>14</td>
      <td>40</td>
      <td>7</td>
      <td>0.842347</td>
      <td>0.003795</td>
      <td>7</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2.607199</td>
      <td>0.058006</td>
      <td>33</td>
      <td>76</td>
      <td>9</td>
      <td>0.839207</td>
      <td>0.003414</td>
      <td>8</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.000058</td>
      <td>0.051218</td>
      <td>7</td>
      <td>18</td>
      <td>5</td>
      <td>0.827605</td>
      <td>0.003070</td>
      <td>9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.000324</td>
      <td>0.039254</td>
      <td>14</td>
      <td>8</td>
      <td>4</td>
      <td>0.814037</td>
      <td>0.002486</td>
      <td>10</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.002164</td>
      <td>0.623764</td>
      <td>25</td>
      <td>17</td>
      <td>2</td>
      <td>0.802681</td>
      <td>0.001737</td>
      <td>11</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.011592</td>
      <td>0.099709</td>
      <td>5</td>
      <td>78</td>
      <td>2</td>
      <td>0.801780</td>
      <td>0.002669</td>
      <td>12</td>
    </tr>
    <tr>
      <th>9</th>
      <td>179.542793</td>
      <td>0.036169</td>
      <td>3</td>
      <td>25</td>
      <td>25</td>
      <td>0.801261</td>
      <td>0.002987</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.337553</td>
      <td>0.008691</td>
      <td>76</td>
      <td>4</td>
      <td>6</td>
      <td>0.799951</td>
      <td>0.001863</td>
      <td>14</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.003627</td>
      <td>0.009312</td>
      <td>7</td>
      <td>35</td>
      <td>10</td>
      <td>0.785018</td>
      <td>0.002276</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>185.383783</td>
      <td>0.011758</td>
      <td>41</td>
      <td>1</td>
      <td>15</td>
      <td>0.777975</td>
      <td>0.002354</td>
      <td>16</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.000001</td>
      <td>0.022697</td>
      <td>2</td>
      <td>24</td>
      <td>2</td>
      <td>0.772924</td>
      <td>0.001723</td>
      <td>17</td>
    </tr>
    <tr>
      <th>19</th>
      <td>56.524961</td>
      <td>2.930116</td>
      <td>8</td>
      <td>2</td>
      <td>218</td>
      <td>0.763806</td>
      <td>0.011950</td>
      <td>18</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.452465</td>
      <td>0.005245</td>
      <td>31</td>
      <td>5</td>
      <td>5</td>
      <td>0.758947</td>
      <td>0.000013</td>
      <td>19</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.025624</td>
      <td>0.002431</td>
      <td>7</td>
      <td>22</td>
      <td>92</td>
      <td>0.758947</td>
      <td>0.000013</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>

```python
df = pd.DataFrame(
    {
        'max_leaf_nodes' : cv_results['max_leaf_nodes'],
        'learning_rate' : cv_results['learning_rate'],
        'score_bin' : pd.cut(cv_results['mean_test_score'], bins=np.linspace(0.5,1.0,6))
    }
)

sns.set_palette('YlGnBu_r')

ax = sns.scatterplot(data=df, x= 'max_leaf_nodes', y='learning_rate', hue='score_bin',color='k', edgecolor=None)

ax.set_xscale('log')
ax.set_yscale('log')

ax.legend(title='mean_test_score',loc='center left' ,bbox_to_anchor = (1,0.5))
```

    <matplotlib.legend.Legend at 0x2a2c6655e50>

![png](output_30_1.png)

```python
import numpy as np
import plotly.express as px

fig = px.parallel_coordinates(cv_results.apply(
    {
        'learning_rate' : np.log10,
        'max_leaf_nodes' : np.log2,
        'max_bins' : np.log2,
        'min_samples_leaf' : np.log10,
        'l2_regularization' : np.log10,
        'mean_test_score' : lambda x : x,
    }
),
color='mean_test_score',
color_continuous_scale = px.colors.sequential.Viridis,
)
fig.show()
```

    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    AttributeError: 'numpy.float64' object has no attribute 'log10'


    The above exception was the direct cause of the following exception:


    TypeError                                 Traceback (most recent call last)

    ~\anaconda3\lib\site-packages\pandas\core\apply.py in agg(self)
       1105             try:
    -> 1106                 result = self.obj.apply(f)
       1107             except (ValueError, AttributeError, TypeError):


    ~\anaconda3\lib\site-packages\pandas\core\series.py in apply(self, func, convert_dtype, args, **kwargs)
       4432         """
    -> 4433         return SeriesApply(self, func, convert_dtype, args, kwargs).apply()
       4434


    ~\anaconda3\lib\site-packages\pandas\core\apply.py in apply(self)
       1081
    -> 1082         return self.apply_standard()
       1083


    ~\anaconda3\lib\site-packages\pandas\core\apply.py in apply_standard(self)
       1123             if isinstance(f, np.ufunc):
    -> 1124                 return f(obj)
       1125


    ~\anaconda3\lib\site-packages\pandas\core\generic.py in __array_ufunc__(self, ufunc, method, *inputs, **kwargs)
       2112     ):
    -> 2113         return arraylike.array_ufunc(self, ufunc, method, *inputs, **kwargs)
       2114


    ~\anaconda3\lib\site-packages\pandas\core\arraylike.py in array_ufunc(self, ufunc, method, *inputs, **kwargs)
        396         inputs = tuple(extract_array(x, extract_numpy=True) for x in inputs)
    --> 397         result = getattr(ufunc, method)(*inputs, **kwargs)
        398     else:


    TypeError: loop of ufunc does not support argument 0 of type numpy.float64 which has no callable log10 method


    During handling of the above exception, another exception occurred:


    AttributeError                            Traceback (most recent call last)

    AttributeError: 'numpy.float64' object has no attribute 'log10'


    The above exception was the direct cause of the following exception:


    TypeError                                 Traceback (most recent call last)

    ~\anaconda3\lib\site-packages\pandas\core\apply.py in agg(self)
        738         try:
    --> 739             result = super().agg()
        740         except TypeError as err:


    ~\anaconda3\lib\site-packages\pandas\core\apply.py in agg(self)
        167         if is_dict_like(arg):
    --> 168             return self.agg_dict_like()
        169         elif is_list_like(arg):


    ~\anaconda3\lib\site-packages\pandas\core\apply.py in agg_dict_like(self)
        474             # key used for column selection and output
    --> 475             results = {
        476                 key: obj._gotitem(key, ndim=1).agg(how) for key, how in arg.items()


    ~\anaconda3\lib\site-packages\pandas\core\apply.py in <dictcomp>(.0)
        475             results = {
    --> 476                 key: obj._gotitem(key, ndim=1).agg(how) for key, how in arg.items()
        477             }


    ~\anaconda3\lib\site-packages\pandas\core\series.py in aggregate(self, func, axis, *args, **kwargs)
       4302         op = SeriesApply(self, func, convert_dtype=False, args=args, kwargs=kwargs)
    -> 4303         result = op.agg()
       4304         return result


    ~\anaconda3\lib\site-packages\pandas\core\apply.py in agg(self)
       1107             except (ValueError, AttributeError, TypeError):
    -> 1108                 result = f(self.obj)
       1109


    ~\anaconda3\lib\site-packages\pandas\core\generic.py in __array_ufunc__(self, ufunc, method, *inputs, **kwargs)
       2112     ):
    -> 2113         return arraylike.array_ufunc(self, ufunc, method, *inputs, **kwargs)
       2114


    ~\anaconda3\lib\site-packages\pandas\core\arraylike.py in array_ufunc(self, ufunc, method, *inputs, **kwargs)
        396         inputs = tuple(extract_array(x, extract_numpy=True) for x in inputs)
    --> 397         result = getattr(ufunc, method)(*inputs, **kwargs)
        398     else:


    TypeError: loop of ufunc does not support argument 0 of type numpy.float64 which has no callable log10 method


    The above exception was the direct cause of the following exception:


    TypeError                                 Traceback (most recent call last)

    <ipython-input-90-93e137df1c73> in <module>
          2 import plotly.express as px
          3
    ----> 4 fig = px.parallel_coordinates(cv_results.rename(shorten_param, axis=1).apply(
          5     {
          6         'learning_rate' : np.log10,


    ~\anaconda3\lib\site-packages\pandas\core\frame.py in apply(self, func, axis, raw, result_type, args, **kwargs)
       8831             kwargs=kwargs,
       8832         )
    -> 8833         return op.apply().__finalize__(self, method="apply")
       8834
       8835     def applymap(


    ~\anaconda3\lib\site-packages\pandas\core\apply.py in apply(self)
        696         # dispatch to agg
        697         if is_list_like(self.f):
    --> 698             return self.apply_multiple()
        699
        700         # all empty


    ~\anaconda3\lib\site-packages\pandas\core\apply.py in apply_multiple(self)
        555             Result when self.f is a list-like or dict-like, None otherwise.
        556         """
    --> 557         return self.obj.aggregate(self.f, self.axis, *self.args, **self.kwargs)
        558
        559     def normalize_dictlike_arg(


    ~\anaconda3\lib\site-packages\pandas\core\frame.py in aggregate(self, func, axis, *args, **kwargs)
       8641
       8642         op = frame_apply(self, func=func, axis=axis, args=args, kwargs=kwargs)
    -> 8643         result = op.agg()
       8644
       8645         if relabeling:


    ~\anaconda3\lib\site-packages\pandas\core\apply.py in agg(self)
        743                 f"incompatible data and dtype: {err}"
        744             )
    --> 745             raise exc from err
        746         finally:
        747             self.obj = obj


    TypeError: DataFrame constructor called with incompatible data and dtype: loop of ufunc does not support argument 0 of type numpy.float64 which has no callable log10 method

we observed that some hyperparameters have very little impact on the cross-validation score,

while others have to be adjusted within a specific range to get models with good predictive accuracy.

### Evaluation and hyperparameter tuning

지금까지는 모델의 hyperparameter를 조정하는 방법 두 가지를 배웠다.

이제는 hyperparameter를 조정하기 전단계인, 여러 후보 모델군 중 가장 적절한 모델을 선정하는 방법에 대해서 살펴본다.

```python
model
```

<style>#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 {color: black;background-color: white;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 pre{padding: 0;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-toggleable {background-color: white;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-estimator:hover {background-color: #d4ebff;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-item {z-index: 1;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-parallel::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 2em;bottom: 0;left: 50%;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-parallel-item {display: flex;flex-direction: column;position: relative;background-color: white;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-parallel-item:only-child::after {width: 0;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;position: relative;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-label label {font-family: monospace;font-weight: bold;background-color: white;display: inline-block;line-height: 1.2em;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-label-container {position: relative;z-index: 2;text-align: center;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397 div.sk-text-repr-fallback {display: none;}</style><div id="sk-1a560d91-51cd-47c6-8d17-bc4c6fecb397" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,

                 ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,
                                   transformers=[(&#x27;categorical&#x27;,
                                                  OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,
                                                                 unknown_value=-1),
                                                  [&#x27;workclass&#x27;, &#x27;education&#x27;,
                                                   &#x27;marital-status&#x27;,
                                                   &#x27;occupation&#x27;, &#x27;relationship&#x27;,
                                                   &#x27;race&#x27;, &#x27;sex&#x27;,
                                                   &#x27;native-country&#x27;])])),
                (&#x27;classifier&#x27;,
                 HistGradientBoostingClassifier(max_leaf_nodes=4,
                                                random_state=42))])</pre><b>Please rerun this cell to show the HTML repr or trust the notebook.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="0627268f-5d09-4e97-a4f8-adf076c6e937" type="checkbox" ><label for="0627268f-5d09-4e97-a4f8-adf076c6e937" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,
                                   transformers=[(&#x27;categorical&#x27;,
                                                  OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,
                                                                 unknown_value=-1),
                                                  [&#x27;workclass&#x27;, &#x27;education&#x27;,
                                                   &#x27;marital-status&#x27;,
                                                   &#x27;occupation&#x27;, &#x27;relationship&#x27;,
                                                   &#x27;race&#x27;, &#x27;sex&#x27;,
                                                   &#x27;native-country&#x27;])])),
                (&#x27;classifier&#x27;,
                 HistGradientBoostingClassifier(max_leaf_nodes=4,
                                                random_state=42))])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="6a838cae-ec94-4911-93fb-f21b4136c670" type="checkbox" ><label for="6a838cae-ec94-4911-93fb-f21b4136c670" class="sk-toggleable__label sk-toggleable__label-arrow">preprocessor: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(remainder=&#x27;passthrough&#x27;, sparse_threshold=0,
                  transformers=[(&#x27;categorical&#x27;,
                                 OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;,
                                                unknown_value=-1),
                                 [&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;,
                                  &#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;,
                                  &#x27;native-country&#x27;])])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="5cc120f0-cf69-4d64-8940-b0be0bc255ee" type="checkbox" ><label for="5cc120f0-cf69-4d64-8940-b0be0bc255ee" class="sk-toggleable__label sk-toggleable__label-arrow">categorical</label><div class="sk-toggleable__content"><pre>[&#x27;workclass&#x27;, &#x27;education&#x27;, &#x27;marital-status&#x27;, &#x27;occupation&#x27;, &#x27;relationship&#x27;, &#x27;race&#x27;, &#x27;sex&#x27;, &#x27;native-country&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="2dfe2d21-7c5b-4a4b-8e70-a8089f90ad8c" type="checkbox" ><label for="2dfe2d21-7c5b-4a4b-8e70-a8089f90ad8c" class="sk-toggleable__label sk-toggleable__label-arrow">OrdinalEncoder</label><div class="sk-toggleable__content"><pre>OrdinalEncoder(handle_unknown=&#x27;use_encoded_value&#x27;, unknown_value=-1)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="5d7afcf5-d5b9-4c71-8f0e-eb98b834f606" type="checkbox" ><label for="5d7afcf5-d5b9-4c71-8f0e-eb98b834f606" class="sk-toggleable__label sk-toggleable__label-arrow">remainder</label><div class="sk-toggleable__content"><pre></pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="995d3baf-150b-4ac1-bb16-1ac8a3b506ab" type="checkbox" ><label for="995d3baf-150b-4ac1-bb16-1ac8a3b506ab" class="sk-toggleable__label sk-toggleable__label-arrow">passthrough</label><div class="sk-toggleable__content"><pre>passthrough</pre></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="aa57b700-7460-460f-9c95-8f5cefb1acc0" type="checkbox" ><label for="aa57b700-7460-460f-9c95-8f5cefb1acc0" class="sk-toggleable__label sk-toggleable__label-arrow">HistGradientBoostingClassifier</label><div class="sk-toggleable__content"><pre>HistGradientBoostingClassifier(max_leaf_nodes=4, random_state=42)</pre></div></div></div></div></div></div></div>

```python
from sklearn.model_selection import cross_validate
cv_result = cross_validate(model, data, target, cv=5)
cv_result = pd.DataFrame(cv_result)
cv_result
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
      <th>fit_time</th>
      <th>score_time</th>
      <th>test_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.842183</td>
      <td>0.060796</td>
      <td>0.863036</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.636012</td>
      <td>0.052124</td>
      <td>0.860784</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.673908</td>
      <td>0.058627</td>
      <td>0.860360</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.562928</td>
      <td>0.069014</td>
      <td>0.863124</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.637467</td>
      <td>0.088637</td>
      <td>0.867219</td>
    </tr>
  </tbody>
</table>
</div>

```python

```
