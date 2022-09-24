---
title: "Numpy 유용한 함수 정리"
category: "dataTool"
date: "2022-08-10"
thumbnail: "./images/numpy.png"
---

## Numpy

### arg = index 관련 매소드

argsort : sort로 분류 한 다음 index 반환

argmin : min 값 index 반환

argmax : max 값 index 반환

### shape 명령어 결과

3차원 =(a,b,c)
2차원 =(a,b)
1차원 =(a,)
scalar = ()

### axis = 0 row, axis =1 column

### np.flatnonzero()

특정 array 중 0이 아닌 값의 index를 반환

pandas filter로 사용해도 되겠다. [0,1,0,1...]인 값들에서 위치만 뽑을 수 있음

```python
x = np.arrange(-2,3)

=> array([-2,-1,0,1,2])

np.flatnonzero(x)
=> array([0,1,3,4])

### pandas filter 사용 예시
misclassified_samples_idx = np.flatnonzero(target != target_predicted)
data_misclassified = data.iloc[misclassified_samples_idx]
```

### np.zeros_like(array, dtype=)

특정 array의 shape를 복사하지만 내용은 0으로 채우는 명령어

유사 명령어로 `np.zero()`가 있음. shape를 원하는데로 설정해야하는 점에서 차이가 있음.

```python
target.shape
=> (342,)

sample_weight = np.zeros_like(target, dtype=int)
=> (342,)
```

### np.intersect1d(a,b)

교집합을 구하는 매소드

### np.linspace(시작, 끝, 개수, dtype=int or float)

```python
numbers = np.linspace(5,50,24,dtype=int) # 5~50 사이를 24개로 균등하게 나눈 뒤 int로 반환
numbers
>>> array([ 5,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
       38, 40, 42, 44, 46, 48, 50])
```

### np.where(조건식, True일 때 수식, False일 때 수식)

- **조건식만 적을 경우 index 반환**

```python
square =  np.array([
        [16,3,2,13],
        [5,10,11,8],
        [9,6,7,12],
        [4,15,14,1],
    ])

# 조건식만 반영한 경우(x축 y축 튜플로 Return함)
a = np.where(square > 3)
a
>>> (array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3], dtype=int64), <= x축 tuple
     array([0, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2], dtype=int64)) <= y축 tuple

### 세로로 읽으면 (0,0), (0,3)... 이 됨  3 이상인 위치 idx를 반환함

pd.DataFrame(a).T ### (x,y) 식으로 변경되서 나옴

# 조건식, 변경식까지 추가한 경우
a = np.where(square % 4 == 0, square*2, square-square)
a
>> array([[32,  0,  0,  0],
	       [ 0,  0,  0, 16],
	       [ 0,  0,  0, 24],
	       [ 8,  0,  0,  0]])
```

### np.flatten()과 np.ravel()

flatten과 ravel은 vector를 1차원 백터로 만든다.

np.flatten은 새로운 값을 만드는 반면에 np.ravel()은 모양만 바꾼다.(겉으로만 바뀐것 처럼 보인다.)

얕은 복사와 깊은 복사 개념을 이야기하는 듯.

### np.in1d(array1, array2) array1 해당 요소가 array2에 있으면 True, 아니면 False

```python
**>>>** values **=** np.array([6,0,0,3,2,5,6])
>>> np.in1d(values, [6,2,5])

array([ **True**, **False**, **False**, **False**, **True**, **True**, **True**])
```

### np.isin(arr1(찾을), arr2(찾고))

```python
from numpy import ndarray
import numpy as np

datas = np.asarray([1,2,3,4,5,6,7])

# 이 값들의 포함 여부를 알려달라.
iwantit = np.asarray([1,4,6,10])

# 해당 ndarray의 index 위치에 포함 여부가 표시된다.
print(np.isin(datas, iwantit))
#[ True False False  True False  True False]

출처: https://mellowlee.tistory.com/entry/numpy-npisin-npwhere-index-찾아보기 [잠토의 잠망경]
```
