---
title: "5.3 Importance of decision tree hyperparameters on generalization "
category: "MachineLearning"
date: "2022-04-13"
thumbnail: "./images/scikit-learn-logo.png"
---

https://inria.github.io/scikit-learn-mooc/python_scripts/trees_hyperparameters.html

helper 설치가 어려우므로 이번 장은 홈페이지를 활용하자.

### Effect of the max_depth parameter

- The hyperparameter max_depth controls the overall complexity of a decision tree.

- This hyperparameter allows to get a trade-off between an under-fitted and over-fitted decision tree.

Max depth 2 => underfit / Max depth 30 => overfit

max_depth is one of the hyperparameters that one should optimize via cross-validation and grid-search.

### Other hyperparameters in decision trees

There is no guarantee that a tree will be symmetrical. Indeed, optimal generalization performance could be reached by growing some of the branches deeper than some others.

The hyperparameters min_samples_leaf, min_samples_split, max_leaf_nodes, or min_impurity_decrease allows growing asymmetric trees and apply a constraint at the leaves or nodes level.

- min_samples_leaf : 개별 leaf별로 최소로 가지고 있어야 하는 sample 개수를 정하는 parameter
