---
title: "Section 2.4_ GPUs"
category: "DeepLearning"
date: "2022-09-19"
thumbnail: "./img/nuromatch.png"
---

```python
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import dataloader
from torchvision.transforms import ToTensor

def set_device():
  """
  Set the device. CUDA if available, CPU otherwise

  Args:
    None

  Returns:
    Nothing
  """
  device = "cuda" if torch.cuda.is_available() else "mps"
  if device != "cuda":
    print("GPU is not enabled in this notebook. \n"
          "If you want to enable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `GPU` from the dropdown menu")
  else:
    print("GPU is enabled in this notebook. \n"
          "If you want to disable it, in the menu under `Runtime` -> \n"
          "`Hardware accelerator.` and select `None` from the dropdown menu")

  return device
```

### mps : Apple의 Metal Performance Shaders

- CPU와 MPS 차이가 단순연산에선 최대 100배 정도 나는것 같습니다.

```python
x = torch.randn(10).to('mps')
print(x.device)
```

    mps:0

```python
DEVICE = set_device()
x = torch.randn(2, 2, device=DEVICE)
print(x.dtype)
print(x.device)

# we can also use the .to() method to change the device a tensor lives on
y = torch.randn(2, 2)
print(f"y before calling to() | device: {y.device} | dtype: {y.type()}")

y = y.to(DEVICE)
print(f"y after calling to() | device: {y.device} | dtype: {y.type()}")
```

    GPU is not enabled in this notebook.
    If you want to enable it, in the menu under `Runtime` ->
    `Hardware accelerator.` and select `GPU` from the dropdown menu
    torch.float32
    mps:0
    y before calling to() | device: cpu | dtype: torch.FloatTensor
    y after calling to() | device: mps:0 | dtype: torch.mps.FloatTensor

```python
dim = 10000
iterations = 1

def simpleFun(dim, device):
  """
  Helper function to check device-compatiblity with computations
  Args:
    dim: Integer
    device: String
      "cpu" or "cuda"
  Returns:
    Nothing.
  """
  # 2D tensor filled with uniform random numbers in [0,1), dim x dim
  x = torch.rand(dim, dim).to(device)
  # 2D tensor filled with uniform random numbers in [0,1), dim x dim
  y = torch.rand_like(x).to(device)
  # 2D tensor filled with the scalar value 2, dim x dim
  z = 2*torch.ones(dim, dim).to(device)

  # elementwise multiplication of x and y
  a = x * y
  # matrix multiplication of x and z
  b = x @ z

  del x
  del y
  del z
  del a
  del b


a = time.time()
simpleFun(dim=5000,device='cpu')
b = time.time()-a
print(b)

a = time.time()
simpleFun(dim=5000,device='mps')
b = time.time()-a
print(b)
```

    2.09808349609375e-05
    34.96109390258789

```python

```
