---
title: "Section 2 Creating Tensors"
category: "DeepLearning"
date: "2022-09-19"
thumbnail: "./img/nuromatch.png"
---

## Pytorch ≑ numpy

numpy를 GPU로 사용한다고 생각하면 된다.

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
```

## Basic of Tensor

```python
a = torch.tensor([0,1,2])

c = np.ones([2,3]) # 1이 들어있는 array 생성

c = torch.tensor(c)


```

    tensor([[1., 1., 1.],
            [1., 1., 1.]], dtype=torch.float64)

```python
c = np.ones([5,3]) # 1이 들어있는 array 생성
t = torch.ones(5,3)
print(f'numpy : {c}')
print(f'tensor : {t}')

```

    numpy : [[1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]
     [1. 1. 1.]]
    tensor : tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])

### Zero와 Empty

- zero = 0이 들어있는 tensor 생성

- empty = 메모리 공간을 할당
- empty가 약간 빠르다고 함.

```python
z = torch.zeros(2)
e = torch.empty(1, 1, 5)
print(f"Tensor z: {z}")
print(f"Tensor e: {e}")
```

    Tensor z: tensor([0., 0.])
    Tensor e: tensor([[[9.1477e-41, 0.0000e+00, 1.1210e-44, 0.0000e+00, 6.6415e-37]]])

```python
a = torch.rand(1,5) # 0 ~ 1 균등한 random
b = torch.randn(1,5) # normal distribution random

print(f"Tensor a: {a}")
print(f"Tensor b: {b}")
```

    Tensor a: tensor([[0.3751, 0.3315, 0.9492, 0.0688, 0.2988]])
    Tensor b: tensor([[-1.6029,  0.9502, -0.8179, -1.1040,  0.2414]])

### arrange와 Linspace

```python
a = torch.arange(0,10,step=1)
b = np.arange(0,10,step=1)

c = torch.linspace(0,5,steps=11)
d = np.linspace(0,5,num=11)

print(f"Tensor a: {a}\n")
print(f"Numpy array b: {b}\n")
print(f"Tensor c: {c}\n")
print(f"Numpy array d: {d}\n")
```

    Tensor a: tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    Numpy array b: [0 1 2 3 4 5 6 7 8 9]

    Tensor c: tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000, 3.5000, 4.0000,
            4.5000, 5.0000])

    Numpy array d: [0.  0.5 1.  1.5 2.  2.5 3.  3.5 4.  4.5 5. ]

### 문제

```python
# A
a = torch.ones(20,21)

# B
z = np.ones([3,4])
z = torch.tensor(z)

#C (a와 구조가 같은데 모두 random으로 채우는 방법)
c = torch.rand_like(a)

#D
d = torch.arange(4,42,step=2)

```

### 결론 : Pytorch ≑ GPU버전 numpy라고 이해하자
