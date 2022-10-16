---
title: "Section 2_ PyTorch AutoGrad"
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

### Automatic Differentiation

- w를 구하는 알고리즘이라고 하는 듯

- AutoGrad is PyTorch’s automatic differentiation engine.

### Section 2.1: Forward Propagation

- 사용자가 함수를 제공하면 pytorch는 해당 식을 graph로 그려서 연산한다.

- `weight`, `biases` = `parameters` = `learnable parameters` = `trainable parameters`

- `requires_grad` : to indicate that a certain tensor contains learnable parameters(Tensor 내 parameter를 포함한다는 말인듯)

  > .requires_grad속성을 True로 설정하면, 그 tensor에서 이뤄진 모든 연산들을 추적(track)하기 시작한다.

- `detach` : Tensor가 기록을 추적하는 것을 중단하려면, .detach()를 호출하여 연산 기록으로부터 분리(detach)하여 이후 연산들이 추적되는 것을 방지할 수 있다.

<br/>

Function 클래스

Autograd 구현에서 매우 중요한 클래스가 하나 더 있다. 바로 Function 클래스이다. Tensor와 Function은 서로 연결되어 있으며, 모든 연산 과정을 부호화(encode)하여 순환하지 않는 그래프(acyclic graph)를 생성한다.

각 tensor는 .grad_fn속성을 갖고 있는데, 이는 Tensor를 생성한 Function을 참조하고 있다.(단, 사용자가 만든 Tensor는 예외로, 이 때 grad_fn은 None)이다.

simple graph => w,b가 주어진 상태에서 loss를 구하는 방법 소개

우리가 궁극적으로 할 것은 w,b를 찾는 일

```python
class SimpleGraph :
    def __init__(self,w,b) :
        # class parameter 정의
        assert isinstance(w,float)
        assert isinstance(b,float)
        self.w = torch.tensor([w],requires_grad=True)
        self.b = torch.tensor([b],requires_grad=True)

    def forward(self,x):
        # loss 계산에 활용될 함수 정의
        assert isinstance(x, torch.Tensor)
        prediction = torch.tanh(self.w*x + self.b)
        return prediction

def sq_loss(y_true, y_prediction) :
    # loss function 정의
    assert isinstance(y_true, torch.Tensor)
    assert isinstance(y_prediction, torch.Tensor)
    loss = (y_true - y_prediction)**2
    return loss

# x값, y값 정의
feature = torch.tensor([1])
target = torch.tensor([7])

# 임의의 w와 b 정의
simple_graph = SimpleGraph(-0.5,0.5)
print(f"initial weight = {simple_graph.w.item()}, "
      f"\ninitial bias = {simple_graph.b.item()}")

# loss function 연산을 위한 prediction 계산
prediction = simple_graph.forward(feature)

# loss 계산
square_loss = sq_loss(target, prediction)

print(f"for x={feature.item()} and y={target.item()}, "
      f"prediction={prediction.item()}, and L2 Loss = {square_loss.item()}")
```

    initial weight = -0.5,
    initial bias = 0.5
    for x=1 and y=7, prediction=0.0, and L2 Loss = 49.0

### Section 2.2: Backward Propagation

- Tensor와 function은 상호 연결되어있고 그래프를 구성한다.

- `grad_fn` 변수는 해당 tensor를 만든 함수를 창조한다.(C의 경우 + 함수가 참조됨을 볼 수 있음)

- 그래서 user가 만든 tensor는 grad_fn이 없다.

```python
a = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([-1.0], requires_grad=True)
c = a + b
print(f'Gradient function = {c.grad_fn}')
```

    Gradient function = <AddBackward0 object at 0x1159cee80>

- 복잡하게 연산 과정을 거친 Tensor는 가장 마지막 연산만 보여준다

```python
print(f'Gradient function for prediction = {prediction.grad_fn}')
print(f'Gradient function for loss = {square_loss.grad_fn}')
```

    Gradient function for prediction = <TanhBackward0 object at 0x1159cee80>
    Gradient function for loss = <PowBackward0 object at 0x1159ce6a0>

###

- Tensor.backward()는 `loss`로 불린다.

- weight 구하는 방법

  $\frac{\partial{loss}}{\partial{w}} = - 2 x (y_t - y_p)(1 - y_p^2)$

  $y_t$ = target || $y_p$ = prediction

- bias 구하는 방법

  $\frac{\partial{loss}}{\partial{b}} = - 2 (y_t - y_p)(1 - y_p^2)$
