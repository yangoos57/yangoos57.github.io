---
title: "Section 2.2_Operations in PyTorch"
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
```

### Tensor operations

- Pointwise : 포인트별로 계산한다. 병렬계산을 의미하는듯

```python
a = torch.ones(5,3)
b = torch.rand(5,3)
c = torch.empty_like(a)
d = torch.empty_like(b)

torch.add(a,b, out=c)
torch.multiply(a,b, out=d)

print(c)
```

    tensor([[1.0807, 1.3384, 1.9716],
            [1.7287, 1.7563, 1.1235],
            [1.6141, 1.7030, 1.7693],
            [1.4831, 1.8150, 1.7789],
            [1.6642, 1.0128, 1.0767]])

```python
x = torch.tensor([1, 2, 4, 8])
y = torch.tensor([1, 2, 3, 4])

# tuple로 되네
x + y, x - y, x * y, x / y, x**y
```

    (tensor([ 2,  4,  7, 12]),
     tensor([0, 0, 1, 4]),
     tensor([ 1,  4, 12, 32]),
     tensor([1.0000, 1.0000, 1.3333, 2.0000]),
     tensor([   1,    4,   64, 4096]))

```python
x = torch.rand(3,3)

print(x)
print('합 : ',x.sum())
print('column 계산 : ',x.sum(axis=0))
print('row 계산 : ',x.sum(axis=1))

print('평균 : ',x.mean())
print('column 평균 : ',x.mean(axis=0))
print('row 평균 : ',x.mean(axis=1))
```

    tensor([[0.5195, 0.9064, 0.3317],
            [0.0469, 0.8761, 0.6372],
            [0.6177, 0.3569, 0.0922]])
    합 :  tensor(4.3847)
    column 계산 :  tensor([1.1841, 2.1395, 1.0611])
    row 계산 :  tensor([1.7577, 1.5602, 1.0668])
    평균 :  tensor(0.4872)
    column 평균 :  tensor([0.3947, 0.7132, 0.3537])
    row 평균 :  tensor([0.5859, 0.5201, 0.3556])

### matrix multiplication(@)와 dot multiplication(•)

(1) matrix multiplication(@)
$\begin{equation}
\textbf{A} =
\begin{bmatrix}2 &4 \\5 & 7
\end{bmatrix}
\begin{bmatrix} 1 &1 \\2 & 3
\end{bmatrix}

- \begin{bmatrix}10 & 10 \\ 12 & 1
  \end{bmatrix}
  \end{equation}$

<br/><br/>

(2) dot multiplication(•)
$\begin{equation}
b = 
\begin{bmatrix} 3 \\ 5 \\ 7
\end{bmatrix} \cdot 
\begin{bmatrix} 2 \\ 4 \\ 8
\end{bmatrix}
\end{equation}$

```python
### 1번 연산

a1 = torch.tensor([[2, 4], [5, 7]])
a2 = torch.tensor([[1, 1], [2, 3]])
a3 = torch.tensor([[10, 10], [12, 1]])

a = torch.matmul(a1,a2) + a3

a
```

    tensor([[20, 24],
            [31, 27]])

```python
### 2번 연산
b1 = torch.tensor([3, 5, 7])
b2 = torch.tensor([2, 4, 8])

# b2.T 할필요 없이 자동으로 연산이 수행 됨
torch.dot(b1,b2)
```

    tensor(82)
