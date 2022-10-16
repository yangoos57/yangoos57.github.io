---
title: "Section 2.3 Manipulating Tensors in Pytorch"
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

### Manipulating tensors : indexing

```python
# 5d tensor
x = torch.rand(2,2,3,4,5)

# 값 선택 방법 indexing으로 가능
x[0]
x[0][0][1][1][3]
```

    tensor(0.1978)

```python
# flatten
torch.flatten(x).shape

# reshape
x.reshape(2,120)

```

    ''

### squeezing tensor and unsqueezing

- 임의로 차원을 늘리고 줄이는데 활용된다.

- 차원을 늘리고 줄이는데 활용되므로 차원 중 1을 넣거나 빼는 기능이다.

```python
x = torch.rand(1,10,1,10)

# 3차원을 2차원으로 축소 1이 사라짐
# 어느 차원에 있는 1을 지울건지 parameter로 선택 가능
print(x.squeeze(2).shape)


# 차원을 한단계 증가
# 어느 차원을 늘릴건지 parameter로 선택 가능
x.unsqueeze(2).shape
```

    torch.Size([1, 10, 10])





    torch.Size([1, 10, 1, 1, 10])

### permutation

차원의 순서를 바꾼다.

```python
x = torch.rand(3,4,5)
x = x.permute(0,2,1) # [3,5,4]
x.shape
```

    torch.Size([3, 5, 4])

### Concatnation

tensor 합치기

```python
x = torch.arange(12,dtype=torch.float32).reshape(3,4)
y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# x,y에 괄호 치는거 잊지 말기
cat_row = torch.cat((x,y),dim=0)
cat_col = torch.cat((x,y),dim=1)

print(cat_row.shape)
print(cat_col.shape)
```

    torch.Size([6, 4])
    torch.Size([3, 8])

### tensor to numpy

```python
x = torch.ones(1,3)
x.numpy()
```

    array([[1., 1., 1.]], dtype=float32)

```python
# scalar to int or float
x = torch.ones(1)

# 1번
x.item()

# 2번
int(x)

#3번
float(x)
```

    1.0

**function 1**
$\begin{equation}
  \text{If }
  A = \begin{bmatrix}
  1 & 1 \\
  1 & 1
  \end{bmatrix}
  \text{and }
  B = \begin{bmatrix}
  1 & 2 & 3 \\
  1 & 2 & 3
  \end{bmatrix}
  \text{ then }
  Out =  \begin{bmatrix}
  2 & 2
  \end{bmatrix} \cdot 12 = \begin{bmatrix}
  24 & 24
  \end{bmatrix}
\end{equation}$

**function 2**
$\begin{equation}
  \text{If }
  C = \begin{bmatrix}
  2 & 3 \\
  -1 & 10
  \end{bmatrix}
  \text{ then }
  Out = \begin{bmatrix}
  0 & 2 \\
  1 & 3 \\
  2 & -1 \\
  3 & 10
  \end{bmatrix}
\end{equation}$

**function3**
$\begin{equation}
  \text{If }
  D = \begin{bmatrix}
  1 & -1 \\
  -1 & 3
  \end{bmatrix}
  \text{and } 
  E = \begin{bmatrix}
  2 & 3 & 0 & 2 \\
  \end{bmatrix}
  \text{ then } 
  Out = \begin{bmatrix}
  3 & 2 \\
  -1 & 5
  \end{bmatrix}
\end{equation}$

**function 4**

$\begin{equation}
  \text{If }
  D = \begin{bmatrix}
  1 & -1 \\
  -1 & 3
  \end{bmatrix}
  \text{and }
  E = \begin{bmatrix}
  2 & 3 & 0  \\
  \end{bmatrix}
  \text{ then }
  Out = \begin{bmatrix}
  1 & -1 & -1 & 3  & 2 & 3 & 0  
  \end{bmatrix}
\end{equation}$

```python
# function 1
a = torch.ones(2,2)
b = torch.tensor([[1,2,3],[1,2,3]])

a = a.sum(dim=1)
b = b.sum()
out = a*b
print('function 1 :',out)

#function 2 singleton?
c = torch.tensor([[2,3],[-1,10]])
c = c.flatten()
idx = torch.tensor([0,1,2,3])
out = torch.cat((c,idx),dim=0)
print('function 2 :',out)

# function2 정답
my_tensor = c.flatten()
# index 뽑아주는 식 없음 그냥 range로 만들면 됨
idx_tensor = torch.arange(0, len(my_tensor))
# unsqueeze와 squeeze를 이해해야할듯
output = torch.cat([idx_tensor.unsqueeze(1), my_tensor.unsqueeze(1)], axis=1)


# function 3
a = torch.tensor([[1,-1],[-1,3]])
b = torch.tensor([2,3,0,2])
b = b.reshape(2,2)
out = a+b
print('function 3 :',out)

# function 4

a = torch.tensor([[1,-1],[-1,3]])
b = torch.tensor([2,3,0])

a = a.flatten()
out = torch.cat((a,b))
print('function 4 :',out)

```

    function 1 : tensor([24., 24.])
    function 2 : tensor([ 2,  3, -1, 10,  0,  1,  2,  3])
    function 3 : tensor([[ 3,  2],
            [-1,  5]])
    function 4 : tensor([ 1, -1, -1,  3,  2,  3,  0])
