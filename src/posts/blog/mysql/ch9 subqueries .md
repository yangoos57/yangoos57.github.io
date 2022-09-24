---
title: "ch9 subqueries"
category: "MySQL"
date: "2022-03-25"
thumbnail: "./mysql.png"
---

```python
import pandas as pd
import pymysql

conn = pymysql.connect(host='localhost', port=int(3306), user='root',passwd='1234', db='sakila')
cursor = conn.cursor(pymysql.cursors.DictCursor)
```

### Subqueries

- a query contained within another SQL statment.
- It is always enclosed within parentheses

```python
sen = '''

SELECT customer_id, first_name, last_name
FROM customer
WHERE customer_id =(SELECT MAX(customer_id) From customer)

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
      <th>customer_id</th>
      <th>first_name</th>
      <th>last_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>599</td>
      <td>AUSTIN</td>
      <td>CINTRON</td>
    </tr>
  </tbody>
</table>
</div>

### Subquery Types

- single row/column
- single row/multicolumn
- multiple column
- noncorrelated subqueries
- correlated subqueries

```python
sen = '''

SELECT city_id, city
FROM city
WHERE country_id <>
    (SELECT country_id FROM country WHERE country ='india')

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
      <th>city_id</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>A Corua (La Corua)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Abha</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Abu Dhabi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Acua</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Adana</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>535</th>
      <td>596</td>
      <td>Zaria</td>
    </tr>
    <tr>
      <th>536</th>
      <td>597</td>
      <td>Zeleznogorsk</td>
    </tr>
    <tr>
      <th>537</th>
      <td>598</td>
      <td>Zhezqazghan</td>
    </tr>
    <tr>
      <th>538</th>
      <td>599</td>
      <td>Zhoushan</td>
    </tr>
    <tr>
      <th>539</th>
      <td>600</td>
      <td>Ziguinchor</td>
    </tr>
  </tbody>
</table>
<p>540 rows × 2 columns</p>
</div>

### Single row/ column이 필요한 이유

WHERE country_id 는 하나의 값만 받을 수 있는데 반환된 값은 하나의 set이기 때문에 오류가 발생함

```python
sen = '''

SELECT city_id, city
FROM city
WHERE country_id <>
    (SELECT country_id FROM country WHERE country !='india')

'''
cursor.execute(sen)
pd.DataFrame(cursor.fetchall())
```

    ---------------------------------------------------------------------------

    OperationalError                          Traceback (most recent call last)

    <ipython-input-5-11499b4897ec> in <module>
          7
          8 '''
    ----> 9 cursor.execute(sen)
         10 pd.DataFrame(cursor.fetchall())


    ~\anaconda3\lib\site-packages\pymysql\cursors.py in execute(self, query, args)
        146         query = self.mogrify(query, args)
        147
    --> 148         result = self._query(query)
        149         self._executed = query
        150         return result


    ~\anaconda3\lib\site-packages\pymysql\cursors.py in _query(self, q)
        308         self._last_executed = q
        309         self._clear_result()
    --> 310         conn.query(q)
        311         self._do_get_result()
        312         return self.rowcount


    ~\anaconda3\lib\site-packages\pymysql\connections.py in query(self, sql, unbuffered)
        546             sql = sql.encode(self.encoding, "surrogateescape")
        547         self._execute_command(COMMAND.COM_QUERY, sql)
    --> 548         self._affected_rows = self._read_query_result(unbuffered=unbuffered)
        549         return self._affected_rows
        550


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _read_query_result(self, unbuffered)
        773         else:
        774             result = MySQLResult(self)
    --> 775             result.read()
        776         self._result = result
        777         if result.server_status is not None:


    ~\anaconda3\lib\site-packages\pymysql\connections.py in read(self)
       1154     def read(self):
       1155         try:
    -> 1156             first_packet = self.connection._read_packet()
       1157
       1158             if first_packet.is_ok_packet():


    ~\anaconda3\lib\site-packages\pymysql\connections.py in _read_packet(self, packet_type)
        723             if self._result is not None and self._result.unbuffered_active is True:
        724                 self._result.unbuffered_active = False
    --> 725             packet.raise_for_error()
        726         return packet
        727


    ~\anaconda3\lib\site-packages\pymysql\protocol.py in raise_for_error(self)
        219         if DEBUG:
        220             print("errno =", errno)
    --> 221         err.raise_mysql_exception(self._data)
        222
        223     def dump(self):


    ~\anaconda3\lib\site-packages\pymysql\err.py in raise_mysql_exception(data)
        141     if errorclass is None:
        142         errorclass = InternalError if errno < 1000 else OperationalError
    --> 143     raise errorclass(errno, errval)


    OperationalError: (1242, 'Subquery returns more than 1 row')

### multi-row, single-column

하나의 set을 활용하는 query는 네 종류가 있다.

1. in and not in operators

```python
sen = '''

SELECT country_id
FROM country
WHERE country IN ('Canada','Mexico');

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
      <th>country_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>

```python
sen = '''

SELECT city_id, city
FROM city
WHERE country_id IN
    (
        SELECT country_id
        FROM country
        WHERE country IN('Canada','Mexico')
    );

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
      <th>city_id</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>179</td>
      <td>Gatineau</td>
    </tr>
    <tr>
      <th>1</th>
      <td>196</td>
      <td>Halifax</td>
    </tr>
    <tr>
      <th>2</th>
      <td>300</td>
      <td>Lethbridge</td>
    </tr>
    <tr>
      <th>3</th>
      <td>313</td>
      <td>London</td>
    </tr>
    <tr>
      <th>4</th>
      <td>383</td>
      <td>Oshawa</td>
    </tr>
    <tr>
      <th>5</th>
      <td>430</td>
      <td>Richmond Hill</td>
    </tr>
    <tr>
      <th>6</th>
      <td>565</td>
      <td>Vancouver</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4</td>
      <td>Acua</td>
    </tr>
    <tr>
      <th>8</th>
      <td>19</td>
      <td>Allende</td>
    </tr>
    <tr>
      <th>9</th>
      <td>40</td>
      <td>Atlixco</td>
    </tr>
    <tr>
      <th>10</th>
      <td>103</td>
      <td>Carmen</td>
    </tr>
    <tr>
      <th>11</th>
      <td>106</td>
      <td>Celaya</td>
    </tr>
    <tr>
      <th>12</th>
      <td>124</td>
      <td>Coacalco de Berriozbal</td>
    </tr>
    <tr>
      <th>13</th>
      <td>125</td>
      <td>Coatzacoalcos</td>
    </tr>
    <tr>
      <th>14</th>
      <td>129</td>
      <td>Cuauhtmoc</td>
    </tr>
    <tr>
      <th>15</th>
      <td>130</td>
      <td>Cuautla</td>
    </tr>
    <tr>
      <th>16</th>
      <td>131</td>
      <td>Cuernavaca</td>
    </tr>
    <tr>
      <th>17</th>
      <td>154</td>
      <td>El Fuerte</td>
    </tr>
    <tr>
      <th>18</th>
      <td>188</td>
      <td>Guadalajara</td>
    </tr>
    <tr>
      <th>19</th>
      <td>202</td>
      <td>Hidalgo</td>
    </tr>
    <tr>
      <th>20</th>
      <td>212</td>
      <td>Huejutla de Reyes</td>
    </tr>
    <tr>
      <th>21</th>
      <td>213</td>
      <td>Huixquilucan</td>
    </tr>
    <tr>
      <th>22</th>
      <td>246</td>
      <td>Jos Azueta</td>
    </tr>
    <tr>
      <th>23</th>
      <td>250</td>
      <td>Jurez</td>
    </tr>
    <tr>
      <th>24</th>
      <td>288</td>
      <td>La Paz</td>
    </tr>
    <tr>
      <th>25</th>
      <td>330</td>
      <td>Matamoros</td>
    </tr>
    <tr>
      <th>26</th>
      <td>335</td>
      <td>Mexicali</td>
    </tr>
    <tr>
      <th>27</th>
      <td>341</td>
      <td>Monclova</td>
    </tr>
    <tr>
      <th>28</th>
      <td>365</td>
      <td>Nezahualcyotl</td>
    </tr>
    <tr>
      <th>29</th>
      <td>393</td>
      <td>Pachuca de Soto</td>
    </tr>
    <tr>
      <th>30</th>
      <td>445</td>
      <td>Salamanca</td>
    </tr>
    <tr>
      <th>31</th>
      <td>451</td>
      <td>San Felipe del Progreso</td>
    </tr>
    <tr>
      <th>32</th>
      <td>452</td>
      <td>San Juan Bautista Tuxtepec</td>
    </tr>
    <tr>
      <th>33</th>
      <td>541</td>
      <td>Torren</td>
    </tr>
    <tr>
      <th>34</th>
      <td>556</td>
      <td>Uruapan</td>
    </tr>
    <tr>
      <th>35</th>
      <td>563</td>
      <td>Valle de Santiago</td>
    </tr>
    <tr>
      <th>36</th>
      <td>595</td>
      <td>Zapopan</td>
    </tr>
  </tbody>
</table>
</div>

```python
sen = '''

SELECT city_id, city
FROM city
WHERE country_id IN
    (
        SELECT country_id
        FROM country
        WHERE country NOT IN('Canada','Mexico')
    );

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
      <th>city_id</th>
      <th>city</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>251</td>
      <td>Kabul</td>
    </tr>
    <tr>
      <th>1</th>
      <td>59</td>
      <td>Batna</td>
    </tr>
    <tr>
      <th>2</th>
      <td>63</td>
      <td>Bchar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>483</td>
      <td>Skikda</td>
    </tr>
    <tr>
      <th>4</th>
      <td>516</td>
      <td>Tafuna</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>558</th>
      <td>455</td>
      <td>Sanaa</td>
    </tr>
    <tr>
      <th>559</th>
      <td>518</td>
      <td>Taizz</td>
    </tr>
    <tr>
      <th>560</th>
      <td>280</td>
      <td>Kragujevac</td>
    </tr>
    <tr>
      <th>561</th>
      <td>368</td>
      <td>Novi Sad</td>
    </tr>
    <tr>
      <th>562</th>
      <td>272</td>
      <td>Kitwe</td>
    </tr>
  </tbody>
</table>
<p>563 rows × 2 columns</p>
</div>

2. The all operator

   `ALL operator` allows you to make comparsions between a single value and every value in a set

```python
sen = '''

SELECT first_name, last_name
FROM customer
WHERE customer_id <> ALL
    (
        SELECT customer_id
        FROM payment
        WHERE amount =0
    );

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MARY</td>
      <td>SMITH</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PATRICIA</td>
      <td>JOHNSON</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LINDA</td>
      <td>WILLIAMS</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BARBARA</td>
      <td>JONES</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ELIZABETH</td>
      <td>BROWN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>571</th>
      <td>TERRENCE</td>
      <td>GUNDERSON</td>
    </tr>
    <tr>
      <th>572</th>
      <td>ENRIQUE</td>
      <td>FORSYTHE</td>
    </tr>
    <tr>
      <th>573</th>
      <td>FREDDIE</td>
      <td>DUGGAN</td>
    </tr>
    <tr>
      <th>574</th>
      <td>WADE</td>
      <td>DELVALLE</td>
    </tr>
    <tr>
      <th>575</th>
      <td>AUSTIN</td>
      <td>CINTRON</td>
    </tr>
  </tbody>
</table>
<p>576 rows × 2 columns</p>
</div>

### NOT IN 또는 <> ALL 사용 시 주의할 점

Null VALUE가 포함되면 return으로 아무것도 반환하지 않는다.

```python
sen = '''

SELECT city_id, city
FROM city
WHERE country_id IN (123, NULL)

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

3. any operator

ALL과 마찬가지로 ANY도 SET의 element를 하나씩 꺼내서 비교한다.

ANY 설명이 좀 빈약하다. 나중에 더 찾아봐야겠다.

### Multicolumn subqueries

```python
sen = '''

SELECT fa.actor_id, fa.film_id
FROM film_actor fa
WHERE fa.actor_id IN
    ( SELECT actor_id FROM actor WHERE last_name = 'MONROE')
    AND fa.film_id IN
    ( SELECT film_id FROM film WHERE rating = 'PG');

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
      <th>actor_id</th>
      <th>film_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>120</td>
      <td>63</td>
    </tr>
    <tr>
      <th>1</th>
      <td>120</td>
      <td>144</td>
    </tr>
    <tr>
      <th>2</th>
      <td>120</td>
      <td>414</td>
    </tr>
    <tr>
      <th>3</th>
      <td>120</td>
      <td>590</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120</td>
      <td>715</td>
    </tr>
    <tr>
      <th>5</th>
      <td>120</td>
      <td>894</td>
    </tr>
    <tr>
      <th>6</th>
      <td>178</td>
      <td>164</td>
    </tr>
    <tr>
      <th>7</th>
      <td>178</td>
      <td>194</td>
    </tr>
    <tr>
      <th>8</th>
      <td>178</td>
      <td>273</td>
    </tr>
    <tr>
      <th>9</th>
      <td>178</td>
      <td>311</td>
    </tr>
    <tr>
      <th>10</th>
      <td>178</td>
      <td>983</td>
    </tr>
  </tbody>
</table>
</div>

### 위 statement와 아래 statment의 결과는 같다.

Multi column을 어떻게 대처하는지 배울 수 있다.

```python
sen = '''

SELECT fa.actor_id, fa.film_id
FROM film_actor fa
WHERE (fa.actor_id, film_id) IN
    (
        SELECT a.actor_id, f.film_id
        FROM actor a
            CROSS JOIN film f
        WHERE a.last_name ='MONROE'
        AND f.rating = 'PG'
    );

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
      <th>actor_id</th>
      <th>film_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>120</td>
      <td>63</td>
    </tr>
    <tr>
      <th>1</th>
      <td>120</td>
      <td>144</td>
    </tr>
    <tr>
      <th>2</th>
      <td>120</td>
      <td>414</td>
    </tr>
    <tr>
      <th>3</th>
      <td>120</td>
      <td>590</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120</td>
      <td>715</td>
    </tr>
    <tr>
      <th>5</th>
      <td>120</td>
      <td>894</td>
    </tr>
    <tr>
      <th>6</th>
      <td>178</td>
      <td>164</td>
    </tr>
    <tr>
      <th>7</th>
      <td>178</td>
      <td>194</td>
    </tr>
    <tr>
      <th>8</th>
      <td>178</td>
      <td>273</td>
    </tr>
    <tr>
      <th>9</th>
      <td>178</td>
      <td>311</td>
    </tr>
    <tr>
      <th>10</th>
      <td>178</td>
      <td>983</td>
    </tr>
  </tbody>
</table>
</div>

### Correlated subqueries

unrelated queries는 최종 테이블을 만들기 전에 실행됨. 반면 correlated queries는 table이 만들어진 후 검색 용도로서 활용됨

correlated subqueries는 row 하나씩 불러온다고 한다. 이 말은 데이터가 클 경우에 오랜 시간이 걸린다는 의미

```python
sen = '''

SELECT c.first_name, c.last_name
FROM customer c
WHERE 20 = (SELECT count(*) FROM rental r
            WHERE r.customer_id = c.customer_id);


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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LAUREN</td>
      <td>HUDSON</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JEANETTE</td>
      <td>GREENE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TARA</td>
      <td>RYAN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>WILMA</td>
      <td>RICHARDS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JO</td>
      <td>FOWLER</td>
    </tr>
    <tr>
      <th>5</th>
      <td>KAY</td>
      <td>CALDWELL</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DANIEL</td>
      <td>CABRAL</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ANTHONY</td>
      <td>SCHWAB</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TERRY</td>
      <td>GRISSOM</td>
    </tr>
    <tr>
      <th>9</th>
      <td>LUIS</td>
      <td>YANEZ</td>
    </tr>
    <tr>
      <th>10</th>
      <td>HERBERT</td>
      <td>KRUGER</td>
    </tr>
    <tr>
      <th>11</th>
      <td>OSCAR</td>
      <td>AQUINO</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RAUL</td>
      <td>FORTIER</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NELSON</td>
      <td>CHRISTENSON</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ALFREDO</td>
      <td>MCADAMS</td>
    </tr>
  </tbody>
</table>
</div>

```python
sen = '''

SELECT c.first_name, c.last_name
FROM customer c
WHERE
    ( SELECT sum(p.amount) FROM payment p
      WHERE p.customer_id = c.customer_id )
      BETWEEN 180 AND 240;


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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RHONDA</td>
      <td>KENNEDY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CLARA</td>
      <td>SHAW</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ELEANOR</td>
      <td>HUNT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MARION</td>
      <td>SNYDER</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TOMMY</td>
      <td>COLLAZO</td>
    </tr>
    <tr>
      <th>5</th>
      <td>KARL</td>
      <td>SEAL</td>
    </tr>
  </tbody>
</table>
</div>

### The exsists operator

correlated subqueries를 효율적으로 활용하는 operator라고 한다. 존재 여부만 체크하는데 사용된다고 한다.

> quantity를 따져야 하는 경우가 아니면 더 빠른 결과를 발생시키는가 봄

```python
sen = '''

SELECT c.first_name, c.last_name
FROM customer c
WHERE EXISTS
    (SELECT 1 FROM rental r
     WHERE r.customer_id = c.customer_id
     AND date(r.rental_date) < '2005-05-25'
     );


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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CHARLOTTE</td>
      <td>HUNTER</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DELORES</td>
      <td>HANSEN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MINNIE</td>
      <td>ROMERO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CASSANDRA</td>
      <td>WALTERS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ANDREW</td>
      <td>PURDY</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MANUEL</td>
      <td>MURRELL</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TOMMY</td>
      <td>COLLAZO</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NELSON</td>
      <td>CHRISTENSON</td>
    </tr>
  </tbody>
</table>
</div>

### When to Use Subqueries

1. Subqueries as Data Sources

   테이블에 없는 데이터를 가공해서 만든 다음 새로운 table에 넣는다.

```python
sen = '''

SELECT c.first_name, c.last_name, pymnt.num_rentals, pymnt.tot_payments
FROM customer c
    INNER JOIN
    (SELECT customer_id, count(*) num_rentals, sum(amount) tot_payments
    FROM payment
    GROUP BY customer_id) pymnt
    ON c.customer_id = pymnt.customer_id;
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
      <th>num_rentals</th>
      <th>tot_payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MARY</td>
      <td>SMITH</td>
      <td>32</td>
      <td>118.68</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PATRICIA</td>
      <td>JOHNSON</td>
      <td>27</td>
      <td>128.73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LINDA</td>
      <td>WILLIAMS</td>
      <td>26</td>
      <td>135.74</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BARBARA</td>
      <td>JONES</td>
      <td>22</td>
      <td>81.78</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ELIZABETH</td>
      <td>BROWN</td>
      <td>38</td>
      <td>144.62</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>594</th>
      <td>TERRENCE</td>
      <td>GUNDERSON</td>
      <td>30</td>
      <td>117.70</td>
    </tr>
    <tr>
      <th>595</th>
      <td>ENRIQUE</td>
      <td>FORSYTHE</td>
      <td>28</td>
      <td>96.72</td>
    </tr>
    <tr>
      <th>596</th>
      <td>FREDDIE</td>
      <td>DUGGAN</td>
      <td>25</td>
      <td>99.75</td>
    </tr>
    <tr>
      <th>597</th>
      <td>WADE</td>
      <td>DELVALLE</td>
      <td>22</td>
      <td>83.78</td>
    </tr>
    <tr>
      <th>598</th>
      <td>AUSTIN</td>
      <td>CINTRON</td>
      <td>19</td>
      <td>83.81</td>
    </tr>
  </tbody>
</table>
<p>599 rows × 4 columns</p>
</div>

2. Data Fabrication

```python

```
