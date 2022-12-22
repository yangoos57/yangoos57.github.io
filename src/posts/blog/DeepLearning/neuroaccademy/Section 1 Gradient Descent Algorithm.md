---
title: "Section 1 Gradient Descent Algorithm"
category: "DeepLearning"
date: "2022-09-18"
thumbnail: "./img/nuromatch.png"
---

## Gradient Descent Algorithm

- 학습 알고리즘의 목표, risk function의 output을 최소화 하는 것(risk = cost = loss )

- gradient descent 수식화

  - Neural Network function : $y=f_w(x)$(w는 weight을 의미)

  - A loss function : $L=∂(y,data)$(y와 data의 차이를 비교)

  - Optimization problem : $w^* = argmin_w∂(f_w(x),data)$

  최적의 w는 NN과 Loss function을 합한 결과 중 최솟값인 경우를 의미함

<br/>

### Gradient와 Gradient Descent의 차이

**gradient Descent**

1847년 Augustin-Louis Cauchy이 negative of gradients라는 개념을 고안했음. gradient는 chain 함수의 최댓값을 구하는 variable을 찾는 방법이었음. 이를 정반대로 한 negative of gradient는 현재 gradient descent라는 용어로 사용되고 있음.

In 1847, Augustin-Louis Cauchy used negative of gradients to develop the Gradient Descent algorithm as an iterative method to minimize a continuous and (ideally) differentiable function of many variables.

**Gradient vector 찾기(공식 유도)**

- 예시 함수 : $z = h(x, y) = \sin(x^2 + y^2)$

- gradient vector 찾기
  $ = \begin{bmatrix}
  \dfrac{\partial z}{\partial x} \\ \\ \dfrac{\partial z}{\partial y}
  \end{bmatrix}$

**Chain Rule 이해하기**

- 기본 룰
  $F(x) = g(h(x)) \equiv (g \circ h)(x)$

- 미분 할 경우
  $F'(x) = g'(h(x)) \cdot h'(x)$

- 그래프 해석
  1. x축,y축은 각각 h(x,y) 값을 나타냄. 색상은 h(x,y)의 값을 표현
  2. ...
     ![as](./img/W1D2_Tutorial1_Solution_115a15ba_0.png)

<br/>

### Gradient Descent 그래프 이해하기

![a](./img/gradient_1.png)

- gradient를 사용하지 않으면 위 그림과 같이 $\theta_0$ 와 $\theta_1$를 격자로 생성해서 loss를 하나하나 대입하여 구해야함.

- 일일이 모든 값을 구한 뒤 최소 loss를 찾는 방식은 매우 비효율적임

<br/>

![a](./img/gradient_2.png)

- 앞서 언급했듯 gradient는 최대값을 찾는 방법이었음. 매 연산 다음 단계의 최적 방향을 알려주기 떄문에 정해진 길을 걸으면 최단경로로 최대 값에 도달할 수 있었음

- 이와 반대 개념인 gradient descent를 활용하면 최소 loss값을 나타내는 $\theta_0$ 와 $\theta_1$를 찾을 수 있음. 이는 Gradient descent는 매번 가장 가파른 하락을 나타내는 방향을 가리키므로, 최소값의 방향으로 점차 수렴하기 때문

<br/>

### Section 1.2: Gradient Descent Algorithm

**Gradient Descent 식** = $
\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla f \left( \mathbf{w}^{(t)} \right)
$

- $\eta$ = learning Rate, &nbsp; w = weight, &nbsp; $\eta \nabla f \left( \mathbf{w}^{(t)} \right)$ = $w^{(t)}$의 미분

- 이때 $\nabla f (\mathbf{w})= \left( \frac{\partial f(\mathbf{w})}{\partial w_1}, ..., \frac{\partial f(\mathbf{w})}{\partial w_d} \right)$ 을 알면 다음 w 값을 계산할 수 있다.

### Computational Graph를 활용해 $\nabla f (\mathbf{w})$ 구해보기

**예시 함수**

$
f(x, y, z) = \tanh \left(\ln \left[1 + z \frac{2x}{sin(y)} \right] \right)
$

**$\nabla f (\mathbf{w})$ 구하기**

$
\dfrac{\partial Loss}{\partial \mathbf{w}} = \left[ \dfrac{\partial Loss}{\partial w_1}, \dfrac{\partial Loss}{\partial w_2} , \dots, \dfrac{\partial Loss}{\partial w_d} \right]^{\top}
$

<br/>

- 우리가 구하고자 하는 값들(1~d) : $\dfrac{\partial Loss}{\partial w_d}$

- $\dfrac{\partial Loss}{\partial w_d}$을 하나씩 구해보자.

### Computational Graph(forward) - 원래 개념

- 위 공식의 계산 과정을 하나하나 뜯어내 도식화 하였음

- x,y,z를 대입해 $f$를 출력 함

  ![a](./img/comput_graph_forward.png)

<br/>

### Computational Graph(backward) - 델타 값 구하기 위해 응용

- $f$ 값을 가지고 $\dfrac{\partial f}{\partial x}$, $\dfrac{\partial f}{\partial y}$, $\dfrac{\partial f}{\partial z}$ 값을 출력함

- 파란색 네모 박스는 미분한 값

  ![a](./img/comput_graph_backward.png)

### 결론

$
\dfrac{\partial f}{\partial x} = \dfrac{\partial f}{\partial e}~\dfrac{\partial e}{\partial d}~\dfrac{\partial d}{\partial c}~\dfrac{\partial c}{\partial a}~\dfrac{\partial a}{\partial x} = \left( 1-\tanh^2(e) \right) \cdot \frac{1}{d+1}\cdot z \cdot \frac{1}{b} \cdot 2
$
