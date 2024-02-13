---
title: "ch5 Querying multiple tables "
category: "MySQL"
date: "2022-03-22"
thumbnail: "./mysql.png"
---

```python
import pandas as pd
import pymysql

conn = pymysql.connect(host='localhost', port=int(3306), user='root',passwd='1234', db='sakila')
cursor = conn.cursor(pymysql.cursors.DictCursor)
```

### JOIN

- ch4에서 INNER와 ON, 그리고 공통의 foreign key를 사용해 여러 테이블의 자료를 불러오는 방법을 배웠다.
- JOIN도 이와 마찬가지로 여러 테이블에 있는 데이터를 하나의 테이블로 만드는 방법이다.
- 하지만 JOIN은 foreign key가 공유되지 않더라도 테이블을 만들 수 있다.

```python
sen = '''

SELECT c.first_name, c.last_name, a.address
FROM customer AS C JOIN address AS a ;

'''

cursor.execute(sen)
pd.DataFrame(cursor.fetchall())
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>first_name</th>
      <th>last_name</th>
      <th>address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AUSTIN</td>
      <td>CINTRON</td>
      <td>47 MySakila Drive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WADE</td>
      <td>DELVALLE</td>
      <td>47 MySakila Drive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FREDDIE</td>
      <td>DUGGAN</td>
      <td>47 MySakila Drive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ENRIQUE</td>
      <td>FORSYTHE</td>
      <td>47 MySakila Drive</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TERRENCE</td>
      <td>GUNDERSON</td>
      <td>47 MySakila Drive</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>361192</th>
      <td>ELIZABETH</td>
      <td>BROWN</td>
      <td>1325 Fukuyama Street</td>
    </tr>
    <tr>
      <th>361193</th>
      <td>BARBARA</td>
      <td>JONES</td>
      <td>1325 Fukuyama Street</td>
    </tr>
    <tr>
      <th>361194</th>
      <td>LINDA</td>
      <td>WILLIAMS</td>
      <td>1325 Fukuyama Street</td>
    </tr>
    <tr>
      <th>361195</th>
      <td>PATRICIA</td>
      <td>JOHNSON</td>
      <td>1325 Fukuyama Street</td>
    </tr>
    <tr>
      <th>361196</th>
      <td>MARY</td>
      <td>SMITH</td>
      <td>1325 Fukuyama Street</td>
    </tr>
  </tbody>
</table>
<p>361197 rows × 3 columns</p>
</div>

### Cartesian product

361197이 나온 이유는 599\*603을 했기 때문임. 이는 어떻게 join할지 명령어를 추가하지 않아서 발생한다.

```python
sen = '''

SELECT c.first_name, c.last_name, a.address
FROM customer AS c JOIN address AS a
    ON c.address_id = a.address_id;

'''

cursor.execute(sen)
pd.DataFrame(cursor.fetchall())
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>first_name</th>
      <th>last_name</th>
      <th>address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MARY</td>
      <td>SMITH</td>
      <td>1913 Hanoi Way</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PATRICIA</td>
      <td>JOHNSON</td>
      <td>1121 Loja Avenue</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LINDA</td>
      <td>WILLIAMS</td>
      <td>692 Joliet Street</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BARBARA</td>
      <td>JONES</td>
      <td>1566 Inegl Manor</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ELIZABETH</td>
      <td>BROWN</td>
      <td>53 Idfu Parkway</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>594</th>
      <td>TERRENCE</td>
      <td>GUNDERSON</td>
      <td>844 Bucuresti Place</td>
    </tr>
    <tr>
      <th>595</th>
      <td>ENRIQUE</td>
      <td>FORSYTHE</td>
      <td>1101 Bucuresti Boulevard</td>
    </tr>
    <tr>
      <th>596</th>
      <td>FREDDIE</td>
      <td>DUGGAN</td>
      <td>1103 Quilmes Boulevard</td>
    </tr>
    <tr>
      <th>597</th>
      <td>WADE</td>
      <td>DELVALLE</td>
      <td>1331 Usak Boulevard</td>
    </tr>
    <tr>
      <th>598</th>
      <td>AUSTIN</td>
      <td>CINTRON</td>
      <td>1325 Fukuyama Street</td>
    </tr>
  </tbody>
</table>
<p>599 rows × 3 columns</p>
</div>

```python
sen = '''

SELECT c.first_name, c.last_name, a.address
FROM customer AS c INNER JOIN address AS a
    ON c.address_id = a.address_id;

'''

cursor.execute(sen)
pd.DataFrame(cursor.fetchall())
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>first_name</th>
      <th>last_name</th>
      <th>address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MARY</td>
      <td>SMITH</td>
      <td>1913 Hanoi Way</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PATRICIA</td>
      <td>JOHNSON</td>
      <td>1121 Loja Avenue</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LINDA</td>
      <td>WILLIAMS</td>
      <td>692 Joliet Street</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BARBARA</td>
      <td>JONES</td>
      <td>1566 Inegl Manor</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ELIZABETH</td>
      <td>BROWN</td>
      <td>53 Idfu Parkway</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>594</th>
      <td>TERRENCE</td>
      <td>GUNDERSON</td>
      <td>844 Bucuresti Place</td>
    </tr>
    <tr>
      <th>595</th>
      <td>ENRIQUE</td>
      <td>FORSYTHE</td>
      <td>1101 Bucuresti Boulevard</td>
    </tr>
    <tr>
      <th>596</th>
      <td>FREDDIE</td>
      <td>DUGGAN</td>
      <td>1103 Quilmes Boulevard</td>
    </tr>
    <tr>
      <th>597</th>
      <td>WADE</td>
      <td>DELVALLE</td>
      <td>1331 Usak Boulevard</td>
    </tr>
    <tr>
      <th>598</th>
      <td>AUSTIN</td>
      <td>CINTRON</td>
      <td>1325 Fukuyama Street</td>
    </tr>
  </tbody>
</table>
<p>599 rows × 3 columns</p>
</div>

### JOIN을 사용할 때 종류를 적어주는 습관을 들이자.

JOIN의 종류는 여러가지가 있지만 이번 장에는 INNER와 OUTTER만 설명하고 있다.

INNER JOIN은 ON으로 매칭되는 값만 반환하는 방법이고

OUTTER JOIN은 매칭되지 않는 값이라 할지라도 모두 반환하는 방법이다.

방법에 차이가 있는만큼 JOIN을 쓸때 어떤 종류인지 적는 습관을 들이자

### USING

테이블에 row를 공유할 수 있는 value가 있을때(foreign key) 복잡한 수식없이 간편하게 연결시켜주는 method이다.

```python
sen = '''

SELECT c.first_name, c.last_name, a.address
FROM customer AS c INNER JOIN address AS a
    USING (address_id);

'''
cursor.execute(sen)
pd.DataFrame(cursor.fetchall())

```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>first_name</th>
      <th>last_name</th>
      <th>address</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MARY</td>
      <td>SMITH</td>
      <td>1913 Hanoi Way</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PATRICIA</td>
      <td>JOHNSON</td>
      <td>1121 Loja Avenue</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LINDA</td>
      <td>WILLIAMS</td>
      <td>692 Joliet Street</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BARBARA</td>
      <td>JONES</td>
      <td>1566 Inegl Manor</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ELIZABETH</td>
      <td>BROWN</td>
      <td>53 Idfu Parkway</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>594</th>
      <td>TERRENCE</td>
      <td>GUNDERSON</td>
      <td>844 Bucuresti Place</td>
    </tr>
    <tr>
      <th>595</th>
      <td>ENRIQUE</td>
      <td>FORSYTHE</td>
      <td>1101 Bucuresti Boulevard</td>
    </tr>
    <tr>
      <th>596</th>
      <td>FREDDIE</td>
      <td>DUGGAN</td>
      <td>1103 Quilmes Boulevard</td>
    </tr>
    <tr>
      <th>597</th>
      <td>WADE</td>
      <td>DELVALLE</td>
      <td>1331 Usak Boulevard</td>
    </tr>
    <tr>
      <th>598</th>
      <td>AUSTIN</td>
      <td>CINTRON</td>
      <td>1325 Fukuyama Street</td>
    </tr>
  </tbody>
</table>
<p>599 rows × 3 columns</p>
</div>

### 3개 JOIN 하기

address table을 중심으로 customer table과 city table이 연결됨

```python
sen = '''

SELECT c.first_name, c.last_name, ct.city
FROM customer AS c
    INNER JOIN address AS a
    ON c.address_id = a.address_id
    INNER JOIN city AS ct
    ON ct.city_id = a.city_id;
'''

cursor.execute(sen)
pd.DataFrame(cursor.fetchall())
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>first_name</th>
      <th>last_name</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MARY</td>
      <td>SMITH</td>
      <td>Sasebo</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PATRICIA</td>
      <td>JOHNSON</td>
      <td>San Bernardino</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LINDA</td>
      <td>WILLIAMS</td>
      <td>Athenai</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BARBARA</td>
      <td>JONES</td>
      <td>Myingyan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ELIZABETH</td>
      <td>BROWN</td>
      <td>Nantou</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>594</th>
      <td>TERRENCE</td>
      <td>GUNDERSON</td>
      <td>Jinzhou</td>
    </tr>
    <tr>
      <th>595</th>
      <td>ENRIQUE</td>
      <td>FORSYTHE</td>
      <td>Patras</td>
    </tr>
    <tr>
      <th>596</th>
      <td>FREDDIE</td>
      <td>DUGGAN</td>
      <td>Sullana</td>
    </tr>
    <tr>
      <th>597</th>
      <td>WADE</td>
      <td>DELVALLE</td>
      <td>Lausanne</td>
    </tr>
    <tr>
      <th>598</th>
      <td>AUSTIN</td>
      <td>CINTRON</td>
      <td>Tieli</td>
    </tr>
  </tbody>
</table>
<p>599 rows × 3 columns</p>
</div>

### SubQuery로 구현하기

```python
sen = '''

SELECT c.first_name, c.last_name, addr.address, addr.city
FROM customer AS c
    INNER JOIN
    (
        SELECT a.address_id, a.address, ct.city
        FROM address AS a
            INNER JOIN city AS ct
            ON a.city_id = ct.city_id
            WHERE a.district = 'California'
    ) AS addr
    ON c.address_id = addr.address_id;

'''

cursor.execute(sen)
pd.DataFrame(cursor.fetchall())
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>first_name</th>
      <th>last_name</th>
      <th>address</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PATRICIA</td>
      <td>JOHNSON</td>
      <td>1121 Loja Avenue</td>
      <td>San Bernardino</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BETTY</td>
      <td>WHITE</td>
      <td>770 Bydgoszcz Avenue</td>
      <td>Citrus Heights</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ALICE</td>
      <td>STEWART</td>
      <td>1135 Izumisano Parkway</td>
      <td>Fontana</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ROSA</td>
      <td>REYNOLDS</td>
      <td>793 Cam Ranh Avenue</td>
      <td>Lancaster</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RENEE</td>
      <td>LANE</td>
      <td>533 al-Ayn Boulevard</td>
      <td>Compton</td>
    </tr>
    <tr>
      <th>5</th>
      <td>KRISTIN</td>
      <td>JOHNSTON</td>
      <td>226 Brest Manor</td>
      <td>Sunnyvale</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CASSANDRA</td>
      <td>WALTERS</td>
      <td>920 Kumbakonam Loop</td>
      <td>Salinas</td>
    </tr>
    <tr>
      <th>7</th>
      <td>JACOB</td>
      <td>LANCE</td>
      <td>1866 al-Qatif Avenue</td>
      <td>El Monte</td>
    </tr>
    <tr>
      <th>8</th>
      <td>RENE</td>
      <td>MCALISTER</td>
      <td>1895 Zhezqazghan Drive</td>
      <td>Garden Grove</td>
    </tr>
  </tbody>
</table>
</div>

```python
sen = '''

SELECT f.title
FROM film AS f
    INNER JOIN film_actor AS fa
    ON f.film_id = fa.film_id
    INNER JOIN actor AS a
    ON fa.actor_id = a.actor_id
WHERE (
    (a.first_name = 'CATE' AND a.last_name ='MCQUEEN') OR
    (a.first_name = 'CUBA' AND a.last_name = 'BIRCH')
    )

'''

cursor.execute(sen)
pd.DataFrame(cursor.fetchall())
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>

### 두 배우 모두 나온 영화를 찾기 위해서는 Table을 두 번 사용해야 한다.

```python
sen = '''

SELECT f.title
FROM film AS f
    INNER JOIN film_actor AS fa1
    ON f.film_id = fa1.film_id
    INNER JOIN actor AS a1
    ON fa1.actor_id = a1.actor_id
    INNER JOIN film_actor AS fa2
    ON f.film_id = fa2.film_id
    INNER JOIN actor AS a2
    ON fa2.actor_id = a2.actor_id
WHERE (
    (a1.first_name = 'CATE' AND a1.last_name ='MCQUEEN') AND
    (a2.first_name = 'CUBA' AND a2.last_name = 'BIRCH')
    )

'''

cursor.execute(sen)
pd.DataFrame(cursor.fetchall())
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BLOOD ARGONAUTS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TOWERS HURRICANE</td>
    </tr>
  </tbody>
</table>
</div>

### EXERCISE

exercise 5-1

1. a
2. ct.city_id

```python
sen = '''

SELECT f.title, a.first_name
FROM film AS f
    INNER JOIN film_actor AS fa
    ON f.film_id = fa.film_id
    INNER JOIN actor AS a
    ON fa.actor_id = a.actor_id
WHERE a.first_name ='JOHN'

'''
cursor.execute(sen)
pd.DataFrame(cursor.fetchall())
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>first_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ALLEY EVOLUTION</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BEVERLY OUTLAW</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CANDLES GRAPES</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CLEOPATRA DEVIL</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>COLOR PHILADELPHIA</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CONQUERER NUTS</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DAUGHTER MADIGAN</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GLEAMING JAWBREAKER</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GOLDMINE TYCOON</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>HOME PITY</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>INTERVIEW LIAISONS</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ISHTAR ROCKETEER</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>JAPANESE RUN</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>JERSEY SASSY</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>LUKE MUMMY</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MILLION ACE</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MONSTER SPARTACUS</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>NAME DETECTIVE</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>NECKLACE OUTBREAK</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>NEWSIES STORY</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>PET HAUNTING</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PIANIST OUTFIELD</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>PINOCCHIO SIMON</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PITTSBURGH HUNCHBACK</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>QUILLS BULL</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>RAGING AIRPLANE</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>ROXANNE REBEL</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>SATISFACTION CONFIDENTIAL</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>SONG HEDWIG</td>
      <td>JOHN</td>
    </tr>
  </tbody>
</table>
</div>

```python
### exercise 5-3 ### 하나의 테이블을 두번 사용하는 방법

sen = '''

SELECT a1.address AS addr1, a2.address AS addr2, a1.city_id
FROM address a1
    INNER JOIN address a2
WHERE (a1.city_id = a2.city_id) AND (a1.address_id != a2.address_id)


'''


cursor.execute(sen)
pd.DataFrame(cursor.fetchall())
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>addr1</th>
      <th>addr2</th>
      <th>city_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>47 MySakila Drive</td>
      <td>23 Workhaven Lane</td>
      <td>300</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28 MySQL Boulevard</td>
      <td>1411 Lillydale Drive</td>
      <td>576</td>
    </tr>
    <tr>
      <th>2</th>
      <td>23 Workhaven Lane</td>
      <td>47 MySakila Drive</td>
      <td>300</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1411 Lillydale Drive</td>
      <td>28 MySQL Boulevard</td>
      <td>576</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1497 Yuzhou Drive</td>
      <td>548 Uruapan Street</td>
      <td>312</td>
    </tr>
    <tr>
      <th>5</th>
      <td>587 Benguela Manor</td>
      <td>43 Vilnius Manor</td>
      <td>42</td>
    </tr>
    <tr>
      <th>6</th>
      <td>548 Uruapan Street</td>
      <td>1497 Yuzhou Drive</td>
      <td>312</td>
    </tr>
    <tr>
      <th>7</th>
      <td>43 Vilnius Manor</td>
      <td>587 Benguela Manor</td>
      <td>42</td>
    </tr>
  </tbody>
</table>
</div>

파이썬 문법으로 해도 좋고 SQL 문법으로 해도 좋다.
지금은 SQL 문법에 익숙해지자.
