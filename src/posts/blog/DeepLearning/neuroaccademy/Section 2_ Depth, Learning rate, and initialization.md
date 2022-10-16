---
title: "Section 2_ Depth, Learning rate, and initialization"
category: "DeepLearning"
date: "2022-09-19"
thumbnail: "./img/nuromatch.png"
---

```python
# Imports
import torch
import numpy as np
from torch import nn
from math import pi
import matplotlib.pyplot as plt

import random
import torch

def set_seed(seed=None, seed_torch=True):
  """
  Function that controls randomness. NumPy and random modules must be imported.

  Args:
    seed : Integer
      A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
      If `True` sets the random seed for pytorch tensors, so pytorch module
      must be imported. Default is `True`.

  Returns:
    Nothing.
  """
  if seed is None:
    seed = np.random.choice(2 ** 32)
  random.seed(seed)
  np.random.seed(seed)
  if seed_torch:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

  print(f'Random seed {seed} has been set.')


# In case that `DataLoader` is used
def seed_worker(worker_id):
  """
  DataLoader will reseed workers following randomness in
  multi-process data loading algorithm.

  Args:
    worker_id: integer
      ID of subprocess to seed. 0 means that
      the data will be loaded in the main process
      Refer: https://pytorch.org/docs/stable/data.html#data-loading-randomness for more details

  Returns:
    Nothing
  """
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)
```

```python
SEED = 2021
set_seed(seed=SEED)
DEVICE = 'mps'
```

    Random seed 2021 has been set.

## Section 2: Depth, Learning rate, and initialization

### Section 2.1: The effect of depth

### Why might depth be useful?

결론 : 필요한건 강조하고 필요 없는건 없애기 위해서

depth seems like magic. Depth can change the functions a network can represent, the way a network learns, and how a network generalizes to unseen data.

Imagine a single input, single output linear network with 50 hidden layers and only one neuron per layer

$ prediction = x \cdot w*1 \cdot w_2 \cdot \cdot \cdot w*{50} $

Ex)

- $w_i = 2 => y_p = 2^{50} \approx 1.1256 \times 10^{15} $

- $w_i = 0.5 => y_p = 0.5^{50} \approx 8.88 \times 10^{-16} $

### Section 2.2: Choosing a learning rate

- Learning Rate = $\eta$

### Section 2.3: Depth vs Learning Rate

hyperparameters interact.

### Section 2.4: Why initialization is important

gradients are multiplied by the current weight at each layer, so the product can vanish or explode. Therefore, weight initialization is a fundamentally important hyperparameter.
