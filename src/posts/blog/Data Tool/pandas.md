---
title: "Pandas 유용한 함수 정리"
category: "dataTool"
date: "2022-08-11"
thumbnail: "./images/pandas.png"
---

## Pandas

### Groupby로 category value 찾기

```python
a.groupby('ISBN')['지역'].apply(list)

| ISBN          |    지 역      |   |
|---------------|--------------|---|
| 9788931465372 | [강서, 양천] |   |
| 9788931465518 | [강서]       |   |
| 9788931465853 | [강서, 양천] |   |
| 9788931555875 | [강서]       |   |
| 9788931556742 | [양천]       |   |
| ...           | ...          |   |
| 9791196395704 | [강서, 양천] |   |
| 9791196395711 | [강서, 양천] |   |
| 9791196395766 | [양천]       |   |
| 9791196795504 | [양천]       |   |
| 9791197529504 | [강서, 양천] |   |
```

### 범위로 Groupby

```json
bins = [-1, 5, 8, 11, 14, 17, 20, 23]
time = ["0-6", "6-9", "9-12", "12-15", "15-18", "18-21", "21-24"]
data = a.groupby(pd.cut(a, bins=bins, labels=time)).size()
```

### 변수 삭제해서 메모리 줄이기

import gc
result = pd.DataFrame([])
del [[result]] # [[]]가 핵심이라고 한다.
gc.collect()

### 상황별 row filter 고르기

매소드 소개

row를 필터링 해야할때 자주쓰는 매소드가 세가지 있다.

1. DataFrame.query()

   1. 장점

      너무 좋은 기능인데 반해 잘 알려지지 않은 것 같다. 사람들이 쓴 코드를 보면 생각보다 쓰는사람이 많이 없다.

      장점으로는

      1. 이 함수를 쓰면 코드가 Boolean Mask보다 훨씬 깨끗하다.

         <코드>

         불린의 경우 0 < x < 1000이고 a >100 인 row를 찾고 싶으면

         asd > & asd < 코드

         반면 query는

         ‘ 0 < x < 1000’로 할 수 있다.

         변수를 둘때도 @만 붙이면 쉽게쓰여서 좋다.

      2. 한번에 여러 values를 검색할 수 있다.
         1. Boolean mask는 하나의 조건문에 하나의 value만 검색 가능하다. 그러다보니 코드가 한도없이 길어질 수 있다. query는 그런거 없이 하나의 column에서 찾고자 하는 value를 list나 array 등으로 묶어주기만 하면 알아서 검색된다.
      3. 검색용으로 훌륭하다.

         아마 원하는 값을 찾는데 가장 빠른 방법이지 않을까 싶다. 써야하는 코드도 적고 쉽게 쓴다.

      4. 직관적으로 좋고 대용량 자료라도 느리지 않다. pandas 공식문서에서도 연산속도를 빠르게 하는 방법중 하나로 query를 추천한다. 당연하지만 for문, iterrow, apply로 필터링을 하는 방법과는 비교도 안된다.

   b. 단점

   1. 직관적이고 쉽게 코드를 짠다는 점이 장점이자 단점이다. 그러다보니 사용할 수 있는 문법이 제한적이다. 물론 제한적이라고 하나 써보면 왠만한 filtering은 query로 처리할 수 있다.

      <aside>
      💡 These operations are supported by **`[pandas.eval()](https://pandas.pydata.org/docs/reference/api/pandas.eval.html#pandas.eval)`**:

      - Arithmetic operations except for the left shift (`<<`) and right shift (`>>`) operators, e.g., `df + 2 * pi / s ** 4 % 42 - the_golden_ratio`
      - Comparison operations, including chained comparisons, e.g., `2 < df < df2`
      - Boolean operations, e.g., `df < df2 and df3 < df4 or not df_bool`
      - `list` and `tuple` literals, e.g., `[1, 2]` or `(1, 2)`
      - Attribute access, e.g., `df.a`
      - Subscript expressions, e.g., `df[0]`
      - Simple variable evaluation, e.g., `pd.eval("df")` (this is not very useful)
      - Math functions: `sin`, `cos`, `exp`, `log`, `expm1`, `log1p`, `sqrt`, `sinh`, `cosh`, `tanh`, `arcsin`, `arccos`, `arctan`, `arccosh`, `arcsinh`, `arctanh`, `abs`, `arctan2` and `log10`.

      This Python syntax is **not** allowed:

      - Expressions
        Function calls other than math functions.
        `is/is not` operations
        `if` expressionslambda expressions
        `list/set/dict`comprehensions
        Literal `dict` and `set` expressions
        `yield` expressions
        Generator expressions
        Boolean expressions consisting of only scalar values
      - Statements
        Neither simple nor compound statements are allowed.
        This includes things like `for, while, and if`.

      출처 : [https://pandas.pydata.org/docs/user_guide/enhancingperf.html#supported-syntax](https://pandas.pydata.org/docs/user_guide/enhancingperf.html#supported-syntax)

      </aside>

   2. for문을 써야하는 경우 query를 추천하지 않는다.

      아무리 빠르다 한들 수천번 loop를 돌리면 다른 방법과 속도 차이를 체감한다. 처리속도 관련해서는 다음 매소드에서 설명한다.

2. boolean masking

   사람들이 필터링을 할 때 가장 먼저 배우는 방법이자 가장 많이 사용하는 방법이다. 이 방법을 쓰는 이유는 매우빠른 처리속도에 있다. 한 번의 필터링이야 지금 설명하고있는 세가지 매소드 중 아무거나 사용해도 체감상 큰 차이가 없다. 하지만 데이터 분석을 하다보면 loop 수천번 반복해서 매번 filtering을 거치는 경우가 있다.

   그럴때 이 방법이 빛을 발휘한다. 소개하는 매소드 중 가장 빠르며 무려 query보다는 3배 이상 빠르다. 아래 그래프는 row가 1200만개인 데이터를 가지고, 300번 반복한 결과이다. 단위는 초당 loop를 처리한 횟수이다. 9라면 1초에 9번 루프를 돌았다는 말이다.

   <그래프 >

   단점

   필터링 조건이 많아질수록 boolean masking이 끝도없이 늘어난다. 불필요한 변수가 많아지고 코드 또한 길어진다. 그러다보니 코드를 해석하기가 어렵다.

3. Dataframe.isin(list())

   isin은 boolean mask가 가지고 있는 하나의 value만 필터링 해야하는 단점을 보완한 메소드이다. query가 가진 장점과 마찬가지로 하나의 column에 있는 여러 개의 value에 대한 필터링을 한줄로 처리할 수 있다. 뿐만아니라 query보다 처리속도가 빠르다. 하지만 하나의 value 만 필터링할땐 boolean masking보다 느리므로 어떤 방식으로 필터링하는지에 따라 선택을 해야한다.

   단점

   isin 역시 boolean mask와 마찬가지로 필터링이 직관적이지 않다는 단점이 있다.

   조건별 추천 필터링

   기본적으로 빠른 속도가 필요하거나 query로 할 수 없는 경우를 제외하고는 query로 row를 filtering 하는 방법을 추천한다. loop 속 필터링이 반복되는 경우 하나의 column에서 몇 개의 value를 찾을 것인지에 따라 boolean mask를 쓰거나 isin을 사용한다.

   표 3x3

   loop x query

   loop 0 value 1 boolean mask

   loop 0 value > 1 isin

### nan 데이터를 다른 dataframe에서 채우기

```python
ID    Cat1    Cat2    Cat3
1     NaN     75      NaN
2     61      NaN     84
3     NaN     NaN     NaN

ID    Cat1    Cat2    Cat3
1     54      NaN     44
2     NaN     38     NaN
3     49      50      53

ID    Cat1    Cat2    Cat3
1     54      75      44
2     61      38      84
3     49      50      53

**셋 중 하나 써보자**
df1.update(df2, raise_conflict=True)
df3 = df1.combine_first(df2)
df1[pd.isnull(df1)] = df2[~pd.isnull(df2)]
```

### 데이터 포멧

![Untitled](images/Untitled.png)

### Category type 만들기 Series.cat

```python
# type set
a.astype('category’)

# convert to codes
a.cat.codes

# conver to categories
df.country.cat.categories[df.country.cat.codes] =>

```

### 날짜 정리 Series.dt.

```python
.dt.date         # YYYY-MM-DD(문자)
.dt.year         # 연(4자리숫자)
.dt.month        # 월(숫자)
.dt.month_name() # 월(문자)
.dt.day          # 일(숫자)
.dt.day_name()   # 요일이름(문자)
.dt.weekday      # 요일이름(codes)

.dt.time         # HH:MM:SS(문자)
.dt.hour         # 시(숫자)
.dt.minute       # 분(숫자)
.dt.second       # 초(숫자)
```

### 텍스트 정리 Series.str.

- replace(’내용’,’변환’)
- contains(’내용’)
- split(’기준’)
- slice

  ```python
  s = pd.Series(["koala", "dog", "chameleon"])
  s
  0        koala
  1          dog
  2    chameleon

  s.str[0:5:3]
  0    kl
  1     d
  2    cm
  dtype: object
  ```

### pd.to_datetime(test.Time.astype(str))

종종 python datetime이 변환이 안되는 오류가 있음. string으로 바꾼다음 다시 to_datetime으로 하면 pandas 용 datetime으로 변환된다.

### pd.date_range(start= , end= , period=)

```python
pd.date_range(start='2018-04-24', end='2018-04-27', periods=6)
=> DatetimeIndex(['2018-04-24 00:00:00', '2018-04-24 14:24:00',
               '2018-04-25 04:48:00', '2018-04-25 19:12:00',
               '2018-04-26 09:36:00', '2018-04-27 00:00:00'],
                 dtype='datetime64[ns]', freq=None)
```

### DataFrame.groupby([pd.Grouper(key=, freq=)])

시간, 날짜 단위로 groupby를 도와주는 method

```python
test_count = test.groupby([pd.Grouper(key='Time',freq='H')]).size()
=> 1~24시간 별 count(*)
```

- **pd.resample**
  index가 time_series인 경우 grouper처럼 사용할 수 있음.

  ```python
  a = pd.date_range(start='2018-04-24', end='2018-04-27', periods=5)
  k = pd.Series(range(len(a)), index=a)
  k.resample('20H').size()

  => 2018-04-24 00:00:00    2
  2018-04-24 20:00:00    1
  2018-04-25 16:00:00    1
  2018-04-26 12:00:00    1
  Freq: 20H, dtype: int64
  ```

### pd.concat([a,b], keys=)

keys를 사용하면 multi_index 효과가 발생함 + 개별 list에 multi_index를 부여함

### Dataframe.set_index([a,b,c], inplace = True)

Set_index로 multi_index를 만들 수 있음.

### pd.DataFrame([a,b]).T

DataFrame은 list를 row로 인식해서 아래로 쌓인다. 만약 list가 column 단위라면 **.T** method를 활용해 쉽게 바꿀 수 있다.

- pd.DataFrame(list(zip(a,b)))
  zip을 활용해 column단위 list를 row 단위 list로 만든다.

### DataFrame.Apply(function, axis=)

Apply 매소드는 row 또는 column을 하나씩 불러온다음 설정한 function을 적용시킨다. 파이썬 map function과 기능이 같지만 Series와 Dataframe에서 사용할 수 없다.

```python
df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
   A  B
0  4  9
1  4  9
2  4  9
---
df.apply(np.sqrt)
     A    B
0  2.0  3.0
1  2.0  3.0
2  2.0  3.0
---
df.apply(np.sum, axis=1)
0    13
1    13
2    13
dtype: int64
```

### DataFrame.pivot_table(index=x축에 들어갈 column, value= y축에 들어갈 column, aggfunc= 원하는 function)

- Groupby와 pivot_table 공통점과 차이점
  - 공통점
    unique membership을 가지고 대상을 종합한다.
  - 차이점
    groupby는 결과값으로 multi-index가 적용된 Series로 반환한다. 반면 pivot_table은 DataFrame을 반환한다.
    groupby는 두 개 이상 column에 대해서 적용 가능하지만 pivot_table은 두 개 column만 사용가능하다.

### DataFrame.plot.bar()

- row를 bar로 column을 legend로 사용한다.

```python
K
         column =     0.01	   0.10	       1.00	    10.00
Culmen Length (mm)	0.467537	1.725882	3.724988	6.580371
Culmen Depth (mm)	-0.002953	-0.286512	-1.096500	-2.491597

k.plot.barh()
k.plot.bar()
```

![Untitled](images/Untitled%201.png)

![Untitled](images/Untitled%202.png)

### DataIndex 다루기

datetime64 row는 dt를 활용해서 개별 날짜를 분할 할 수 있음.

```python
air_df['month'] = air_df['DateTime'].dt.month
air_df['weekday'] = air_df['DateTime'].dt.day_name()
```

### True 또는 false 개수 세기

```python
a = [0,1,2,3,4,5,3,3,3]
b = a.isin(3)
False, False, False,  True ... True, True

sum(b) # True 개수
len(a) - sumb(b) # False 개수
```

### DataFrame에서 nan 찾기

```python
 pd.isna(cat_data['Cabin'])
```

### True 또는 False 1,0으로 바꾸기

```python
pd.isna(cat_data['Cabin']).astype(int)
```

### pd.eval()

eval은 text를 가지고 명령어를 수행하도록 하는 함수임. syntax가 한정되어있으니 유의해가며 사용하자.

query는 eval + loc로 동작한다고 함.

- 지원하는 syntax
  [https://pandas.pydata.org/docs/user_guide/enhancingperf.html#supported-syntax](https://pandas.pydata.org/docs/user_guide/enhancingperf.html#supported-syntax)

```python
df = pd.DataFrame(np.random.randn(5, 2), columns=["a", "b"])

df.eval("a + b")
Out[23]:
0   -0.246747
1    0.867786
2   -1.626063
3   -1.134978
4   -1.027798
dtype: float64
-----------------

df.eval('C = A + B', inplace=True)
df
>>>   A   B   C
		0  1  10  11
		1  2   8  10
		2  3   6   9
		3  4   4   8
		4  5   2   7
```

## 기타

### 할당된 메모리 없애기

```python
del [[df_1,df_2]]
gc.collect()
df_1=pd.DataFrame()
df_2=pd.DataFrame()
```
