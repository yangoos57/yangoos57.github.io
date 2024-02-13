---
title: "Ch3 Query Primer"
category: "MySQL"
date: "2022-03-20"
thumbnail: "./mysql.png"
---

```python
import pandas as pd
import pymysql
conn=pymysql.connect(host='localhost',port=int(3306),user='root',passwd='1234',db='sakila')
cursor = conn.cursor(pymysql.cursors.DictCursor)
```

### 앞으로 사용할 sakila data 목록 확인

```python
sen = '''

show tables;

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
      <th>Tables_in_sakila</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>actor</td>
    </tr>
    <tr>
      <th>1</th>
      <td>actor_info</td>
    </tr>
    <tr>
      <th>2</th>
      <td>address</td>
    </tr>
    <tr>
      <th>3</th>
      <td>category</td>
    </tr>
    <tr>
      <th>4</th>
      <td>city</td>
    </tr>
    <tr>
      <th>5</th>
      <td>country</td>
    </tr>
    <tr>
      <th>6</th>
      <td>customer</td>
    </tr>
    <tr>
      <th>7</th>
      <td>customer_list</td>
    </tr>
    <tr>
      <th>8</th>
      <td>film</td>
    </tr>
    <tr>
      <th>9</th>
      <td>film_actor</td>
    </tr>
    <tr>
      <th>10</th>
      <td>film_category</td>
    </tr>
    <tr>
      <th>11</th>
      <td>film_list</td>
    </tr>
    <tr>
      <th>12</th>
      <td>film_text</td>
    </tr>
    <tr>
      <th>13</th>
      <td>inventory</td>
    </tr>
    <tr>
      <th>14</th>
      <td>language</td>
    </tr>
    <tr>
      <th>15</th>
      <td>nicer_but_slower_film_list</td>
    </tr>
    <tr>
      <th>16</th>
      <td>payment</td>
    </tr>
    <tr>
      <th>17</th>
      <td>person</td>
    </tr>
    <tr>
      <th>18</th>
      <td>rental</td>
    </tr>
    <tr>
      <th>19</th>
      <td>sales_by_film_category</td>
    </tr>
    <tr>
      <th>20</th>
      <td>sales_by_store</td>
    </tr>
    <tr>
      <th>21</th>
      <td>staff</td>
    </tr>
    <tr>
      <th>22</th>
      <td>staff_list</td>
    </tr>
    <tr>
      <th>23</th>
      <td>store</td>
    </tr>
  </tbody>
</table>
</div>

```python
sen = '''

DESC customer

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
      <th>Field</th>
      <th>Type</th>
      <th>Null</th>
      <th>Key</th>
      <th>Default</th>
      <th>Extra</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>customer_id</td>
      <td>smallint unsigned</td>
      <td>NO</td>
      <td>PRI</td>
      <td>None</td>
      <td>auto_increment</td>
    </tr>
    <tr>
      <th>1</th>
      <td>store_id</td>
      <td>tinyint unsigned</td>
      <td>NO</td>
      <td>MUL</td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>first_name</td>
      <td>varchar(45)</td>
      <td>NO</td>
      <td></td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>last_name</td>
      <td>varchar(45)</td>
      <td>NO</td>
      <td>MUL</td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>email</td>
      <td>varchar(50)</td>
      <td>YES</td>
      <td></td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>5</th>
      <td>address_id</td>
      <td>smallint unsigned</td>
      <td>NO</td>
      <td>MUL</td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>6</th>
      <td>active</td>
      <td>tinyint(1)</td>
      <td>NO</td>
      <td></td>
      <td>1</td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>create_date</td>
      <td>datetime</td>
      <td>NO</td>
      <td></td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td>last_update</td>
      <td>timestamp</td>
      <td>YES</td>
      <td></td>
      <td>CURRENT_TIMESTAMP</td>
      <td>DEFAULT_GENERATED on update CURRENT_TIMESTAMP</td>
    </tr>
  </tbody>
</table>
</div>

### Select Clause & Column Aliases

```python
### column 하나가지고 새로운 컬럼을 만들 수 있다.
### Column Aliases는 기존 table에 없던 새로운 column을 의미한다. as가 포함된 column이 이에 해당한다.

sen = '''

SELECT language_id,
    'COMMON' AS language_usage,
    language_id * 3.141592 AS language_pi_value,
    upper(name) AS language_name
FROM language;

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
      <th>language_id</th>
      <th>language_usage</th>
      <th>language_pi_value</th>
      <th>language_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>COMMON</td>
      <td>3.141592</td>
      <td>ENGLISH</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>COMMON</td>
      <td>6.283184</td>
      <td>ITALIAN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>COMMON</td>
      <td>9.424776</td>
      <td>JAPANESE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>COMMON</td>
      <td>12.566368</td>
      <td>MANDARIN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>COMMON</td>
      <td>15.707960</td>
      <td>FRENCH</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>COMMON</td>
      <td>18.849552</td>
      <td>GERMAN</td>
    </tr>
  </tbody>
</table>
</div>

```python
sen = '''

SELECT version(),
    user(),
    database();

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
      <th>version()</th>
      <th>user()</th>
      <th>database()</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.0.28</td>
      <td>root@localhost</td>
      <td>sakila</td>
    </tr>
  </tbody>
</table>
</div>

### Removing Duplicates => pandas에서 unique method와 같다.

```python
sen = '''
SELECT actor_id FROM film_actor ORDER BY actor_id;
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>5457</th>
      <td>200</td>
    </tr>
    <tr>
      <th>5458</th>
      <td>200</td>
    </tr>
    <tr>
      <th>5459</th>
      <td>200</td>
    </tr>
    <tr>
      <th>5460</th>
      <td>200</td>
    </tr>
    <tr>
      <th>5461</th>
      <td>200</td>
    </tr>
  </tbody>
</table>
<p>5462 rows × 1 columns</p>
</div>

SELECT 뒤에 DISTINCT를 추가하면 Distinct Set을 구할 수 있음.

```python
sen = '''
SELECT DISTINCT actor_id FROM film_actor ORDER BY actor_id;
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>196</td>
    </tr>
    <tr>
      <th>196</th>
      <td>197</td>
    </tr>
    <tr>
      <th>197</th>
      <td>198</td>
    </tr>
    <tr>
      <th>198</th>
      <td>199</td>
    </tr>
    <tr>
      <th>199</th>
      <td>200</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 1 columns</p>
</div>

### From Clause

From은 테이블 여러 개를 불러 올 수 있다. 그중에 마음에 드는 columns을 불러와서 새로운 table을 만들 수 있다.

**Table 종류**

- Permanent tables => database에 저장된 테이블
- Derived tables
- Temporary tables
- Virtual tables

**Derived tables**

```python
sen = '''
SELECT concat(cust.last_name, ',', cust.first_name) full_name
FROM
    (SELECT first_name, last_name, email
     FROM customer
     WHERE first_name = 'JESSIE'
    ) cust;
'''
### Fullname으로 column alias(가명이라는 의미)를 만든다.
### cust라는 table은 parenthese의 내용으로 만들어진다. email은 왜 넣은거지


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
      <th>full_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BANKS,JESSIE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MILAM,JESSIE</td>
    </tr>
  </tbody>
</table>
</div>

**Temporary tables**

```python
sen = '''
CREATE TEMPORARY TABLE actors_j
    (
        actor_id smallint(5),
        first_name varchar(45),
        last_name varchar(45)
    );
'''
### Fullname으로 column alias(가명이라는 의미)를 만든다.
### cust라는 table은 parenthese의 내용으로 만들어진다. email은 왜 넣은거지


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

```python
sen = '''
INSERT INTO actors_j
SELECT actor_id, first_name, last_name
FROM actor
WHERE last_name LIKE 'J%';
'''
### actors_j라는 테이블에 actor라는 테이블에 있는 j가 맨앞에 있는 성을 가진 항목의 actor_id, first_name, last_name을 삽입하라.


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

```python
sen = '''
SELECT * FROM actors_j ORDER BY last_name
'''
### actors_j라는 테이블에 actor라는 테이블에 있는 j가 맨앞에 있는 성을 가진 항목의 actor_id, first_name, last_name을 삽입하라.


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
      <th>first_name</th>
      <th>last_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>119</td>
      <td>WARREN</td>
      <td>JACKMAN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>131</td>
      <td>JANE</td>
      <td>JACKMAN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>MATTHEW</td>
      <td>JOHANSSON</td>
    </tr>
    <tr>
      <th>3</th>
      <td>64</td>
      <td>RAY</td>
      <td>JOHANSSON</td>
    </tr>
    <tr>
      <th>4</th>
      <td>146</td>
      <td>ALBERT</td>
      <td>JOHANSSON</td>
    </tr>
    <tr>
      <th>5</th>
      <td>82</td>
      <td>WOODY</td>
      <td>JOLIE</td>
    </tr>
    <tr>
      <th>6</th>
      <td>43</td>
      <td>KIRK</td>
      <td>JOVOVICH</td>
    </tr>
  </tbody>
</table>
</div>

**Views**
테이블 같은 기능을 하지만 연관된 database는 없다고 한다.

virtual table이라 해도 상관없다.

- Views are created for various reasons, including to hide columns from users and to simplify complex database designs

=> 어떤 이유가 있어서 쓰는데 그 이유가 납득이 되지는 않는 것 같다.

```python
sen = '''
CREATE VIEW cust_vw AS
SELECT customer_id, first_name, last_name, active
FROM customer ;
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

```python
sen = '''
SELECT first_name, last_name
FROM cust_vw
WHERE active = 0 ;
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
      <td>SANDRA</td>
      <td>MARTIN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JUDITH</td>
      <td>COX</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SHEILA</td>
      <td>WELLS</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ERICA</td>
      <td>MATTHEWS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HEIDI</td>
      <td>LARSON</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PENNY</td>
      <td>NEAL</td>
    </tr>
    <tr>
      <th>6</th>
      <td>KENNETH</td>
      <td>GOODEN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>HARRY</td>
      <td>ARCE</td>
    </tr>
    <tr>
      <th>8</th>
      <td>NATHAN</td>
      <td>RUNYON</td>
    </tr>
    <tr>
      <th>9</th>
      <td>THEODORE</td>
      <td>CULP</td>
    </tr>
    <tr>
      <th>10</th>
      <td>MAURICE</td>
      <td>CRAWLEY</td>
    </tr>
    <tr>
      <th>11</th>
      <td>BEN</td>
      <td>EASTER</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CHRISTIAN</td>
      <td>JUNG</td>
    </tr>
    <tr>
      <th>13</th>
      <td>JIMMIE</td>
      <td>EGGLESTON</td>
    </tr>
    <tr>
      <th>14</th>
      <td>TERRANCE</td>
      <td>ROUSH</td>
    </tr>
  </tbody>
</table>
</div>

**Table Link**

다른 테이블에 있는 자료를 불러올 수 있다.
다만 customer_id와 같이 테이블 간 서로 공유하는 column(primery_key..?)가 있어야 한다.

```python
sen = '''

SELECT customer.first_name, customer.last_name,
        time(rental.rental_date) AS rental_time
FROM customer
    INNER JOIN rental
    ON customer.customer_id = rental.customer_id
WHERE date(rental.rental_date) ='2005-06-14';

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
      <th>rental_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JEFFERY</td>
      <td>PINSON</td>
      <td>0 days 22:53:33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ELMER</td>
      <td>NOE</td>
      <td>0 days 22:55:13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MINNIE</td>
      <td>ROMERO</td>
      <td>0 days 23:00:34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MIRIAM</td>
      <td>MCKINNEY</td>
      <td>0 days 23:07:08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DANIEL</td>
      <td>CABRAL</td>
      <td>0 days 23:09:38</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TERRANCE</td>
      <td>ROUSH</td>
      <td>0 days 23:12:46</td>
    </tr>
    <tr>
      <th>6</th>
      <td>JOYCE</td>
      <td>EDWARDS</td>
      <td>0 days 23:16:26</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GWENDOLYN</td>
      <td>MAY</td>
      <td>0 days 23:16:27</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CATHERINE</td>
      <td>CAMPBELL</td>
      <td>0 days 23:17:03</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MATTHEW</td>
      <td>MAHAN</td>
      <td>0 days 23:25:58</td>
    </tr>
    <tr>
      <th>10</th>
      <td>HERMAN</td>
      <td>DEVORE</td>
      <td>0 days 23:35:09</td>
    </tr>
    <tr>
      <th>11</th>
      <td>AMBER</td>
      <td>DIXON</td>
      <td>0 days 23:42:56</td>
    </tr>
    <tr>
      <th>12</th>
      <td>TERRENCE</td>
      <td>GUNDERSON</td>
      <td>0 days 23:47:35</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SONIA</td>
      <td>GREGORY</td>
      <td>0 days 23:50:11</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CHARLES</td>
      <td>KOWALSKI</td>
      <td>0 days 23:54:34</td>
    </tr>
    <tr>
      <th>15</th>
      <td>JEANETTE</td>
      <td>GREENE</td>
      <td>0 days 23:54:46</td>
    </tr>
  </tbody>
</table>
</div>

**Defining Table Aliases**

Table link 예시처럼 두 개 이상의 테이블이 합쳐질때 where, orderby 같은 절이 어느 table에 해당되는지를 정의해야한다.

```python
sen = '''
SELECT c.first_name, c.last_name,
    time(r.rental_date) AS rental_time
From customer AS C
    INNER JOIN rental AS r
    ON c.customer_id = r.customer_id
WHERE date(r.rental_date) = '2005-06-14';
'''
```

### The Where Clause

- The where clause is the mechanism for filtering out unwanted rows from your result set

```python
sen = '''

SELECT  title
FROM film
WHERE rating = 'G' AND rental_duration >=7;

'''
cursor.execute(sen)
# pd.DataFrame(cursor.fetchall())
```

    29

```python
sen = '''

SELECT  title
FROM film
WHERE (rating = 'G' AND rental_duration >=7) OR (rating = 'PG-13' AND rental_duration < 4) ;

'''
cursor.execute(sen)
# pd.DataFrame(cursor.fetchall())
```

    68

### The group by and having clauses

having = where 이라고 하는데 뭐가 다르려나.

```python
sen = '''

SELECT c.first_name, c.last_name, count(*)
FROM customer AS c
    INNER JOIN rental AS r
    ON c.customer_id = r.customer_id
GROUP BY c.first_name, c.last_name
HAVING count(*) >= 40;

'''
### first_name과 last_name이 중복된 개수를 카운트라하는 말인 듯


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
      <th>count(*)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TAMMY</td>
      <td>SANDERS</td>
      <td>41</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CLARA</td>
      <td>SHAW</td>
      <td>42</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ELEANOR</td>
      <td>HUNT</td>
      <td>46</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SUE</td>
      <td>PETERS</td>
      <td>40</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MARCIA</td>
      <td>DEAN</td>
      <td>42</td>
    </tr>
    <tr>
      <th>5</th>
      <td>WESLEY</td>
      <td>BULL</td>
      <td>40</td>
    </tr>
    <tr>
      <th>6</th>
      <td>KARL</td>
      <td>SEAL</td>
      <td>45</td>
    </tr>
  </tbody>
</table>
</div>

```python
sen = '''

SELECT *
FROM customer

'''
### first_name과 last_name이 중복된 개수를 카운트라하는 말인 듯
### Customer ID로 개수를 세는게 더 맞지 않나?

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
      <th>store_id</th>
      <th>first_name</th>
      <th>last_name</th>
      <th>email</th>
      <th>address_id</th>
      <th>active</th>
      <th>create_date</th>
      <th>last_update</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>MARY</td>
      <td>SMITH</td>
      <td>MARY.SMITH@sakilacustomer.org</td>
      <td>5</td>
      <td>1</td>
      <td>2006-02-14 22:04:36</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>PATRICIA</td>
      <td>JOHNSON</td>
      <td>PATRICIA.JOHNSON@sakilacustomer.org</td>
      <td>6</td>
      <td>1</td>
      <td>2006-02-14 22:04:36</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>LINDA</td>
      <td>WILLIAMS</td>
      <td>LINDA.WILLIAMS@sakilacustomer.org</td>
      <td>7</td>
      <td>1</td>
      <td>2006-02-14 22:04:36</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2</td>
      <td>BARBARA</td>
      <td>JONES</td>
      <td>BARBARA.JONES@sakilacustomer.org</td>
      <td>8</td>
      <td>1</td>
      <td>2006-02-14 22:04:36</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>ELIZABETH</td>
      <td>BROWN</td>
      <td>ELIZABETH.BROWN@sakilacustomer.org</td>
      <td>9</td>
      <td>1</td>
      <td>2006-02-14 22:04:36</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>594</th>
      <td>595</td>
      <td>1</td>
      <td>TERRENCE</td>
      <td>GUNDERSON</td>
      <td>TERRENCE.GUNDERSON@sakilacustomer.org</td>
      <td>601</td>
      <td>1</td>
      <td>2006-02-14 22:04:37</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>595</th>
      <td>596</td>
      <td>1</td>
      <td>ENRIQUE</td>
      <td>FORSYTHE</td>
      <td>ENRIQUE.FORSYTHE@sakilacustomer.org</td>
      <td>602</td>
      <td>1</td>
      <td>2006-02-14 22:04:37</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>596</th>
      <td>597</td>
      <td>1</td>
      <td>FREDDIE</td>
      <td>DUGGAN</td>
      <td>FREDDIE.DUGGAN@sakilacustomer.org</td>
      <td>603</td>
      <td>1</td>
      <td>2006-02-14 22:04:37</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>597</th>
      <td>598</td>
      <td>1</td>
      <td>WADE</td>
      <td>DELVALLE</td>
      <td>WADE.DELVALLE@sakilacustomer.org</td>
      <td>604</td>
      <td>1</td>
      <td>2006-02-14 22:04:37</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>598</th>
      <td>599</td>
      <td>2</td>
      <td>AUSTIN</td>
      <td>CINTRON</td>
      <td>AUSTIN.CINTRON@sakilacustomer.org</td>
      <td>605</td>
      <td>1</td>
      <td>2006-02-14 22:04:37</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
  </tbody>
</table>
<p>599 rows × 9 columns</p>
</div>

### The order by Clause

```python
sen = '''

SELECT customer.first_name, customer.last_name,
        time(rental.rental_date) AS rental_time
FROM customer
    INNER JOIN rental
    ON customer.customer_id = rental.customer_id
WHERE date(rental.rental_date) ='2005-06-14'
ORDER BY customer.last_name ;

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
      <th>rental_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DANIEL</td>
      <td>CABRAL</td>
      <td>0 days 23:09:38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CATHERINE</td>
      <td>CAMPBELL</td>
      <td>0 days 23:17:03</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HERMAN</td>
      <td>DEVORE</td>
      <td>0 days 23:35:09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AMBER</td>
      <td>DIXON</td>
      <td>0 days 23:42:56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JOYCE</td>
      <td>EDWARDS</td>
      <td>0 days 23:16:26</td>
    </tr>
    <tr>
      <th>5</th>
      <td>JEANETTE</td>
      <td>GREENE</td>
      <td>0 days 23:54:46</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SONIA</td>
      <td>GREGORY</td>
      <td>0 days 23:50:11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TERRENCE</td>
      <td>GUNDERSON</td>
      <td>0 days 23:47:35</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CHARLES</td>
      <td>KOWALSKI</td>
      <td>0 days 23:54:34</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MATTHEW</td>
      <td>MAHAN</td>
      <td>0 days 23:25:58</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GWENDOLYN</td>
      <td>MAY</td>
      <td>0 days 23:16:27</td>
    </tr>
    <tr>
      <th>11</th>
      <td>MIRIAM</td>
      <td>MCKINNEY</td>
      <td>0 days 23:07:08</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ELMER</td>
      <td>NOE</td>
      <td>0 days 22:55:13</td>
    </tr>
    <tr>
      <th>13</th>
      <td>JEFFERY</td>
      <td>PINSON</td>
      <td>0 days 22:53:33</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MINNIE</td>
      <td>ROMERO</td>
      <td>0 days 23:00:34</td>
    </tr>
    <tr>
      <th>15</th>
      <td>TERRANCE</td>
      <td>ROUSH</td>
      <td>0 days 23:12:46</td>
    </tr>
  </tbody>
</table>
</div>

### multi sort 가능!

```python
sen = '''

SELECT customer.first_name, customer.last_name,
        time(rental.rental_date) AS rental_time
FROM customer
    INNER JOIN rental
    ON customer.customer_id = rental.customer_id
WHERE date(rental.rental_date) ='2005-06-14'
ORDER BY customer.last_name, customer.first_name ;

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
      <th>rental_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DANIEL</td>
      <td>CABRAL</td>
      <td>0 days 23:09:38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CATHERINE</td>
      <td>CAMPBELL</td>
      <td>0 days 23:17:03</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HERMAN</td>
      <td>DEVORE</td>
      <td>0 days 23:35:09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AMBER</td>
      <td>DIXON</td>
      <td>0 days 23:42:56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JOYCE</td>
      <td>EDWARDS</td>
      <td>0 days 23:16:26</td>
    </tr>
    <tr>
      <th>5</th>
      <td>JEANETTE</td>
      <td>GREENE</td>
      <td>0 days 23:54:46</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SONIA</td>
      <td>GREGORY</td>
      <td>0 days 23:50:11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TERRENCE</td>
      <td>GUNDERSON</td>
      <td>0 days 23:47:35</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CHARLES</td>
      <td>KOWALSKI</td>
      <td>0 days 23:54:34</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MATTHEW</td>
      <td>MAHAN</td>
      <td>0 days 23:25:58</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GWENDOLYN</td>
      <td>MAY</td>
      <td>0 days 23:16:27</td>
    </tr>
    <tr>
      <th>11</th>
      <td>MIRIAM</td>
      <td>MCKINNEY</td>
      <td>0 days 23:07:08</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ELMER</td>
      <td>NOE</td>
      <td>0 days 22:55:13</td>
    </tr>
    <tr>
      <th>13</th>
      <td>JEFFERY</td>
      <td>PINSON</td>
      <td>0 days 22:53:33</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MINNIE</td>
      <td>ROMERO</td>
      <td>0 days 23:00:34</td>
    </tr>
    <tr>
      <th>15</th>
      <td>TERRANCE</td>
      <td>ROUSH</td>
      <td>0 days 23:12:46</td>
    </tr>
  </tbody>
</table>
</div>

### Ascending vs Descending sort order

order by 뒤에 `desc`, `asc`를 추가하면 된다.

```python
sen = '''

SELECT customer.first_name, customer.last_name,
        time(rental.rental_date) AS rental_time
FROM customer
    INNER JOIN rental
    ON customer.customer_id = rental.customer_id
WHERE date(rental.rental_date) ='2005-06-14'
ORDER BY time(rental.rental_date) desc;

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
      <th>rental_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JEANETTE</td>
      <td>GREENE</td>
      <td>0 days 23:54:46</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CHARLES</td>
      <td>KOWALSKI</td>
      <td>0 days 23:54:34</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SONIA</td>
      <td>GREGORY</td>
      <td>0 days 23:50:11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TERRENCE</td>
      <td>GUNDERSON</td>
      <td>0 days 23:47:35</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AMBER</td>
      <td>DIXON</td>
      <td>0 days 23:42:56</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HERMAN</td>
      <td>DEVORE</td>
      <td>0 days 23:35:09</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MATTHEW</td>
      <td>MAHAN</td>
      <td>0 days 23:25:58</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CATHERINE</td>
      <td>CAMPBELL</td>
      <td>0 days 23:17:03</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GWENDOLYN</td>
      <td>MAY</td>
      <td>0 days 23:16:27</td>
    </tr>
    <tr>
      <th>9</th>
      <td>JOYCE</td>
      <td>EDWARDS</td>
      <td>0 days 23:16:26</td>
    </tr>
    <tr>
      <th>10</th>
      <td>TERRANCE</td>
      <td>ROUSH</td>
      <td>0 days 23:12:46</td>
    </tr>
    <tr>
      <th>11</th>
      <td>DANIEL</td>
      <td>CABRAL</td>
      <td>0 days 23:09:38</td>
    </tr>
    <tr>
      <th>12</th>
      <td>MIRIAM</td>
      <td>MCKINNEY</td>
      <td>0 days 23:07:08</td>
    </tr>
    <tr>
      <th>13</th>
      <td>MINNIE</td>
      <td>ROMERO</td>
      <td>0 days 23:00:34</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ELMER</td>
      <td>NOE</td>
      <td>0 days 22:55:13</td>
    </tr>
    <tr>
      <th>15</th>
      <td>JEFFERY</td>
      <td>PINSON</td>
      <td>0 days 22:53:33</td>
    </tr>
  </tbody>
</table>
</div>

### Sorting via Numeric placeholders

order by할 때 이름 말고 numeric placeholder로도 가능함.

```python
sen = '''

SELECT c.first_name, c.last_name,
        r.rental_date AS rental_time
FROM customer AS c
    INNER JOIN rental AS r
    ON c.customer_id = r.customer_id
WHERE date(r.rental_date) = '2005-06-14'
ORDER BY 3 desc

'''
cursor.execute(sen)
# pd.DataFrame(cursor.fetchall())
```

    16

### Exercise

```python
### Exercise 3-1
sen = '''

SELECT actor_id, first_name, last_name
FROM actor
ORDER BY last_name, first_name ;
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
      <th>first_name</th>
      <th>last_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>CHRISTIAN</td>
      <td>AKROYD</td>
    </tr>
    <tr>
      <th>1</th>
      <td>182</td>
      <td>DEBBIE</td>
      <td>AKROYD</td>
    </tr>
    <tr>
      <th>2</th>
      <td>92</td>
      <td>KIRSTEN</td>
      <td>AKROYD</td>
    </tr>
    <tr>
      <th>3</th>
      <td>118</td>
      <td>CUBA</td>
      <td>ALLEN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>145</td>
      <td>KIM</td>
      <td>ALLEN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>13</td>
      <td>UMA</td>
      <td>WOOD</td>
    </tr>
    <tr>
      <th>196</th>
      <td>63</td>
      <td>CAMERON</td>
      <td>WRAY</td>
    </tr>
    <tr>
      <th>197</th>
      <td>111</td>
      <td>CAMERON</td>
      <td>ZELLWEGER</td>
    </tr>
    <tr>
      <th>198</th>
      <td>186</td>
      <td>JULIA</td>
      <td>ZELLWEGER</td>
    </tr>
    <tr>
      <th>199</th>
      <td>85</td>
      <td>MINNIE</td>
      <td>ZELLWEGER</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 3 columns</p>
</div>

```python
### Exercise 3-2
sen = '''

SELECT actor_id, first_name, last_name
FROM actor
WHERE last_name = 'WILLIAMS' or last_name = 'DAVIS'

'''

### WHERE last_name IN ('WILLIAMS','DAVIS') IN을 쓰면 되는군..

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
      <th>first_name</th>
      <th>last_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>JENNIFER</td>
      <td>DAVIS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>101</td>
      <td>SUSAN</td>
      <td>DAVIS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>110</td>
      <td>SUSAN</td>
      <td>DAVIS</td>
    </tr>
    <tr>
      <th>3</th>
      <td>172</td>
      <td>GROUCHO</td>
      <td>WILLIAMS</td>
    </tr>
    <tr>
      <th>4</th>
      <td>137</td>
      <td>MORGAN</td>
      <td>WILLIAMS</td>
    </tr>
    <tr>
      <th>5</th>
      <td>72</td>
      <td>SEAN</td>
      <td>WILLIAMS</td>
    </tr>
  </tbody>
</table>
</div>

```python
### Exercise 3-3

sen = '''

SELECT DISTINCT customer_id
FROM rental
WHERE date(rental_date) = '2005-07-05' ;

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>111</td>
    </tr>
    <tr>
      <th>4</th>
      <td>114</td>
    </tr>
    <tr>
      <th>5</th>
      <td>138</td>
    </tr>
    <tr>
      <th>6</th>
      <td>142</td>
    </tr>
    <tr>
      <th>7</th>
      <td>169</td>
    </tr>
    <tr>
      <th>8</th>
      <td>242</td>
    </tr>
    <tr>
      <th>9</th>
      <td>295</td>
    </tr>
    <tr>
      <th>10</th>
      <td>296</td>
    </tr>
    <tr>
      <th>11</th>
      <td>298</td>
    </tr>
    <tr>
      <th>12</th>
      <td>322</td>
    </tr>
    <tr>
      <th>13</th>
      <td>348</td>
    </tr>
    <tr>
      <th>14</th>
      <td>349</td>
    </tr>
    <tr>
      <th>15</th>
      <td>369</td>
    </tr>
    <tr>
      <th>16</th>
      <td>382</td>
    </tr>
    <tr>
      <th>17</th>
      <td>397</td>
    </tr>
    <tr>
      <th>18</th>
      <td>421</td>
    </tr>
    <tr>
      <th>19</th>
      <td>476</td>
    </tr>
    <tr>
      <th>20</th>
      <td>490</td>
    </tr>
    <tr>
      <th>21</th>
      <td>520</td>
    </tr>
    <tr>
      <th>22</th>
      <td>536</td>
    </tr>
    <tr>
      <th>23</th>
      <td>553</td>
    </tr>
    <tr>
      <th>24</th>
      <td>565</td>
    </tr>
    <tr>
      <th>25</th>
      <td>586</td>
    </tr>
    <tr>
      <th>26</th>
      <td>594</td>
    </tr>
  </tbody>
</table>
</div>

```python
### Exercise 3-4

sen = '''

SELECT c.email, r.return_date
FROM customer c
    INNER JOIN rental r
    ON c.customer_id = r.customer_id
WHERE date(r.rental_date) = '2005-06-14'
ORDER BY 2 desc;

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
      <th>email</th>
      <th>return_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DANIEL.CABRAL@sakilacustomer.org</td>
      <td>2005-06-23 22:00:38</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TERRANCE.ROUSH@sakilacustomer.org</td>
      <td>2005-06-23 21:53:46</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MIRIAM.MCKINNEY@sakilacustomer.org</td>
      <td>2005-06-21 17:12:08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GWENDOLYN.MAY@sakilacustomer.org</td>
      <td>2005-06-20 02:40:27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JEANETTE.GREENE@sakilacustomer.org</td>
      <td>2005-06-19 23:26:46</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HERMAN.DEVORE@sakilacustomer.org</td>
      <td>2005-06-19 03:20:09</td>
    </tr>
    <tr>
      <th>6</th>
      <td>JEFFERY.PINSON@sakilacustomer.org</td>
      <td>2005-06-18 21:37:33</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MATTHEW.MAHAN@sakilacustomer.org</td>
      <td>2005-06-18 05:18:58</td>
    </tr>
    <tr>
      <th>8</th>
      <td>MINNIE.ROMERO@sakilacustomer.org</td>
      <td>2005-06-18 01:58:34</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SONIA.GREGORY@sakilacustomer.org</td>
      <td>2005-06-17 21:44:11</td>
    </tr>
    <tr>
      <th>10</th>
      <td>TERRENCE.GUNDERSON@sakilacustomer.org</td>
      <td>2005-06-17 05:28:35</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ELMER.NOE@sakilacustomer.org</td>
      <td>2005-06-17 02:11:13</td>
    </tr>
    <tr>
      <th>12</th>
      <td>JOYCE.EDWARDS@sakilacustomer.org</td>
      <td>2005-06-16 21:00:26</td>
    </tr>
    <tr>
      <th>13</th>
      <td>AMBER.DIXON@sakilacustomer.org</td>
      <td>2005-06-16 04:02:56</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CHARLES.KOWALSKI@sakilacustomer.org</td>
      <td>2005-06-16 02:26:34</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CATHERINE.CAMPBELL@sakilacustomer.org</td>
      <td>2005-06-15 20:43:03</td>
    </tr>
  </tbody>
</table>
</div>

```python

```
