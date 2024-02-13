---
title: "ch4 Filtering"
category: "MySQL"
date: "2022-03-21"
thumbnail: "./mysql.png"
---

```python
import pandas as pd
import pymysql

conn = pymysql.connect(host='localhost', port=int(3306), user='root',passwd='1234', db='sakila')
cursor = conn.cursor(pymysql.cursors.DictCursor)
```

### Using the not Operater

Not operator를 사용하려면

WHERE **NOT** (first_name = 'STEVEN' OR last_name = 'YOUNG') and create_date > '2006-01-01' 예시처럼 맨 앞에 사용해야함

```python
sen = '''
SELECT *
FROM customer
    INNER JOIN rental
    ON customer.customer_id = rental.customer_id
WHERE NOT (first_name = 'STEVEN' OR last_name = 'YOUNG') AND create_date > '2006-01-01'

'''

cursor.execute(sen)
```

    15983

### Range Conditions

```python
sen = '''

SELECT customer_id, rental_date
FROM rental
WHERE rental_date <= '2005-06-16'
    AND rental_date >= '2005-06-14';

'''

### 두 sen은 같은 결과를 보여준다.


sen = '''

SELECT customer_id, rental_date
FROM rental
WHERE rental_date BETWEEN '2005-06-14' AND '2005-06-16';

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
      <th>rental_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>416</td>
      <td>2005-06-14 22:53:33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>516</td>
      <td>2005-06-14 22:55:13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>239</td>
      <td>2005-06-14 23:00:34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>285</td>
      <td>2005-06-14 23:07:08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>310</td>
      <td>2005-06-14 23:09:38</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>359</th>
      <td>148</td>
      <td>2005-06-15 23:20:26</td>
    </tr>
    <tr>
      <th>360</th>
      <td>237</td>
      <td>2005-06-15 23:36:37</td>
    </tr>
    <tr>
      <th>361</th>
      <td>155</td>
      <td>2005-06-15 23:55:27</td>
    </tr>
    <tr>
      <th>362</th>
      <td>341</td>
      <td>2005-06-15 23:57:20</td>
    </tr>
    <tr>
      <th>363</th>
      <td>149</td>
      <td>2005-06-15 23:58:53</td>
    </tr>
  </tbody>
</table>
<p>364 rows × 2 columns</p>
</div>

**String ranges**

String으로 범위를 설정할 수 있다.

> date 나 numerical value로 range를 설정할때는 설정한 범위까지 포함한다.
>
> 하지만 String은 마지막 범위를 포함하지 않는다.
> 예시를 보면 'FR'이 포함되지 않음을 알 수 있다.
>
> 이를 보완하기 위해서는 'FRZ'로 설정해야한다.
>
> >

```python
sen = '''

SELECT last_name, first_name
FROM customer
WHERE last_name BETWEEN 'FA' AND 'FR' ;

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
      <th>last_name</th>
      <th>first_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FARNSWORTH</td>
      <td>JOHN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FENNELL</td>
      <td>ALEXANDER</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FERGUSON</td>
      <td>BERTHA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FERNANDEZ</td>
      <td>MELINDA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FIELDS</td>
      <td>VICKI</td>
    </tr>
    <tr>
      <th>5</th>
      <td>FISHER</td>
      <td>CINDY</td>
    </tr>
    <tr>
      <th>6</th>
      <td>FLEMING</td>
      <td>MYRTLE</td>
    </tr>
    <tr>
      <th>7</th>
      <td>FLETCHER</td>
      <td>MAE</td>
    </tr>
    <tr>
      <th>8</th>
      <td>FLORES</td>
      <td>JULIA</td>
    </tr>
    <tr>
      <th>9</th>
      <td>FORD</td>
      <td>CRYSTAL</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FORMAN</td>
      <td>MICHEAL</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FORSYTHE</td>
      <td>ENRIQUE</td>
    </tr>
    <tr>
      <th>12</th>
      <td>FORTIER</td>
      <td>RAUL</td>
    </tr>
    <tr>
      <th>13</th>
      <td>FORTNER</td>
      <td>HOWARD</td>
    </tr>
    <tr>
      <th>14</th>
      <td>FOSTER</td>
      <td>PHYLLIS</td>
    </tr>
    <tr>
      <th>15</th>
      <td>FOUST</td>
      <td>JACK</td>
    </tr>
    <tr>
      <th>16</th>
      <td>FOWLER</td>
      <td>JO</td>
    </tr>
    <tr>
      <th>17</th>
      <td>FOX</td>
      <td>HOLLY</td>
    </tr>
  </tbody>
</table>
</div>

### Membership Conditions

범위가 아니고 특정 value만 고르는 조건을 말함

> 다중 조건을 걸려면 IN ()을 쓰자

```python
sen = '''

SELECT title, rating
FROM film
WHERE rating ='G' OR rating = 'PG';

'''


### 두 조건의 결과는 같다.

sen = '''

SELECT title, rating
FROM film
WHERE rating IN ('G', 'PG')

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
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACADEMY DINOSAUR</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACE GOLDFINGER</td>
      <td>G</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AFFAIR PREJUDICE</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AFRICAN EGG</td>
      <td>G</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AGENT TRUMAN</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>367</th>
      <td>WON DARES</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>368</th>
      <td>WONDERLAND CHRISTMAS</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>369</th>
      <td>WORDS HUNTER</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>370</th>
      <td>WORST BANGER</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>371</th>
      <td>YOUNG LANGUAGE</td>
      <td>G</td>
    </tr>
  </tbody>
</table>
<p>372 rows × 2 columns</p>
</div>

### Sub Querie를 써서 membership을 찾을 수 있다.

```python
sen = '''

SELECT rating
FROM film
WHERE title LIKE '%PET%';

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
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>G</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PG</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PG</td>
    </tr>
  </tbody>
</table>
</div>

```python
sen = '''
SELECT title, rating
FROM film
WHERE rating IN(
                SELECT rating
                FROM film
                WHERE title LIKE '%PET%');
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
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACADEMY DINOSAUR</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACE GOLDFINGER</td>
      <td>G</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AFFAIR PREJUDICE</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AFRICAN EGG</td>
      <td>G</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AGENT TRUMAN</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>367</th>
      <td>WON DARES</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>368</th>
      <td>WONDERLAND CHRISTMAS</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>369</th>
      <td>WORDS HUNTER</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>370</th>
      <td>WORST BANGER</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>371</th>
      <td>YOUNG LANGUAGE</td>
      <td>G</td>
    </tr>
  </tbody>
</table>
<p>372 rows × 2 columns</p>
</div>

```python
sen = '''
SELECT title, rating
FROM film
WHERE rating NOT IN ('PG-13','R', 'NC-17');
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
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACADEMY DINOSAUR</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACE GOLDFINGER</td>
      <td>G</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AFFAIR PREJUDICE</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AFRICAN EGG</td>
      <td>G</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AGENT TRUMAN</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>367</th>
      <td>WON DARES</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>368</th>
      <td>WONDERLAND CHRISTMAS</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>369</th>
      <td>WORDS HUNTER</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>370</th>
      <td>WORST BANGER</td>
      <td>PG</td>
    </tr>
    <tr>
      <th>371</th>
      <td>YOUNG LANGUAGE</td>
      <td>G</td>
    </tr>
  </tbody>
</table>
<p>372 rows × 2 columns</p>
</div>

### Matching Conditions

left(string, 위치) = '원하는 값'

**wildcard character**

`_` : Exactly one character  
`%` : Any number of characters(including 0)

<br><br>

**Wildcard Character 예시**

F% : String beginning with F

%t : String ending with t

%bas% : Strings containing the substring 'bas'

\_ _ t _ : Four-character strings with t in the third position

```python
sen = '''
SELECT last_name, first_name
FROM customer
WHERE left(last_name,1) = 'Q' ;
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
      <th>last_name</th>
      <th>first_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>QUALLS</td>
      <td>STEPHEN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>QUINTANILLA</td>
      <td>ROGER</td>
    </tr>
    <tr>
      <th>2</th>
      <td>QUIGLEY</td>
      <td>TROY</td>
    </tr>
  </tbody>
</table>
</div>

```python
sen = '''
SELECT last_name, first_name
FROM customer
WHERE last_name LIKE '_A_T%S';
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
      <th>last_name</th>
      <th>first_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MATTHEWS</td>
      <td>ERICA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>WALTERS</td>
      <td>CASSANDRA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WATTS</td>
      <td>SHELLY</td>
    </tr>
  </tbody>
</table>
</div>

### Null value

- Not applicable : 불필요한 경우..?

- Value not yet known : 아직 정보가 입력되지 않은 경우
- Value undefined : 아직 완성되지 않은 경우

> <br>
>
> ⚠️ null value를 다룰 때 조심해야할 사항
>
> 세 종류의 null은 같은 null이 아니다. 구분해야한다.
>
> <br>

```python
sen = '''

SELECT rental_id, customer_id, return_date
FROM rental
WHERE return_date IS NULL;

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
      <th>rental_id</th>
      <th>customer_id</th>
      <th>return_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11496</td>
      <td>155</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11541</td>
      <td>335</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11563</td>
      <td>83</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11577</td>
      <td>219</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11593</td>
      <td>99</td>
      <td>None</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>178</th>
      <td>15862</td>
      <td>215</td>
      <td>None</td>
    </tr>
    <tr>
      <th>179</th>
      <td>15867</td>
      <td>505</td>
      <td>None</td>
    </tr>
    <tr>
      <th>180</th>
      <td>15875</td>
      <td>41</td>
      <td>None</td>
    </tr>
    <tr>
      <th>181</th>
      <td>15894</td>
      <td>168</td>
      <td>None</td>
    </tr>
    <tr>
      <th>182</th>
      <td>15966</td>
      <td>374</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>183 rows × 3 columns</p>
</div>

### A = NULL을 사용하면 결과가 나오지 않음. IS NULL을 사용해야함!

```python
sen = '''

SELECT rental_id, customer_id, return_date
FROM rental
WHERE return_date = NULL;

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

SELECT rental_id, customer_id, return_date
FROM rental
WHERE return_date IS NOT NULL;

'''
cursor.execute(sen)
a = pd.DataFrame(cursor.fetchall())
```

```python
sen = '''

SELECT rental_id, customer_id, return_date
FROM rental
WHERE return_date IS NULL OR return_date NOT BETWEEN '2005-05-01' AND '2005-09-01';

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
      <th>rental_id</th>
      <th>customer_id</th>
      <th>return_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11496</td>
      <td>155</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11541</td>
      <td>335</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11563</td>
      <td>83</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11577</td>
      <td>219</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11593</td>
      <td>99</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>240</th>
      <td>16005</td>
      <td>466</td>
      <td>2005-09-02 02:35:22</td>
    </tr>
    <tr>
      <th>241</th>
      <td>16020</td>
      <td>311</td>
      <td>2005-09-01 18:17:33</td>
    </tr>
    <tr>
      <th>242</th>
      <td>16033</td>
      <td>226</td>
      <td>2005-09-01 02:36:15</td>
    </tr>
    <tr>
      <th>243</th>
      <td>16037</td>
      <td>45</td>
      <td>2005-09-01 02:48:04</td>
    </tr>
    <tr>
      <th>244</th>
      <td>16040</td>
      <td>195</td>
      <td>2005-09-02 02:19:33</td>
    </tr>
  </tbody>
</table>
<p>245 rows × 3 columns</p>
</div>

### Null value는 의식하지 않으면 드러나지 않으니 새로운 데이터베이스를 만질 때 null 여부를 확인하자

### Exercise

Exercise 4-1
101과 107 return

Exercise 4-2
109

```python
## Exercise4-3

sen ='''

SELECT *
FROM payment
WHERE amount IN (1.98, 7.98, 9.98)

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
      <th>payment_id</th>
      <th>customer_id</th>
      <th>staff_id</th>
      <th>rental_id</th>
      <th>amount</th>
      <th>payment_date</th>
      <th>last_update</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1482</td>
      <td>53</td>
      <td>2</td>
      <td>11657</td>
      <td>7.98</td>
      <td>2006-02-14 15:16:03</td>
      <td>2006-02-15 22:12:42</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1670</td>
      <td>60</td>
      <td>2</td>
      <td>12489</td>
      <td>9.98</td>
      <td>2006-02-14 15:16:03</td>
      <td>2006-02-15 22:12:45</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2901</td>
      <td>107</td>
      <td>1</td>
      <td>13079</td>
      <td>1.98</td>
      <td>2006-02-14 15:16:03</td>
      <td>2006-02-15 22:13:03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4234</td>
      <td>155</td>
      <td>2</td>
      <td>11496</td>
      <td>7.98</td>
      <td>2006-02-14 15:16:03</td>
      <td>2006-02-15 22:13:33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4449</td>
      <td>163</td>
      <td>2</td>
      <td>11754</td>
      <td>7.98</td>
      <td>2006-02-14 15:16:03</td>
      <td>2006-02-15 22:13:38</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7243</td>
      <td>267</td>
      <td>2</td>
      <td>12066</td>
      <td>7.98</td>
      <td>2006-02-14 15:16:03</td>
      <td>2006-02-15 22:15:06</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9585</td>
      <td>354</td>
      <td>1</td>
      <td>12759</td>
      <td>7.98</td>
      <td>2006-02-14 15:16:03</td>
      <td>2006-02-15 22:16:47</td>
    </tr>
  </tbody>
</table>
</div>

```python
### Exercise 4-4

sen ='''

SELECT *
FROM customer
WHERE last_name LIKE '_%AW%'  ;

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
      <td>132</td>
      <td>2</td>
      <td>ESTHER</td>
      <td>CRAWFORD</td>
      <td>ESTHER.CRAWFORD@sakilacustomer.org</td>
      <td>136</td>
      <td>1</td>
      <td>2006-02-14 22:04:36</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>144</td>
      <td>1</td>
      <td>CLARA</td>
      <td>SHAW</td>
      <td>CLARA.SHAW@sakilacustomer.org</td>
      <td>148</td>
      <td>1</td>
      <td>2006-02-14 22:04:36</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>159</td>
      <td>1</td>
      <td>JILL</td>
      <td>HAWKINS</td>
      <td>JILL.HAWKINS@sakilacustomer.org</td>
      <td>163</td>
      <td>1</td>
      <td>2006-02-14 22:04:36</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>192</td>
      <td>1</td>
      <td>LAURIE</td>
      <td>LAWRENCE</td>
      <td>LAURIE.LAWRENCE@sakilacustomer.org</td>
      <td>196</td>
      <td>1</td>
      <td>2006-02-14 22:04:36</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>200</td>
      <td>2</td>
      <td>JEANNE</td>
      <td>LAWSON</td>
      <td>JEANNE.LAWSON@sakilacustomer.org</td>
      <td>204</td>
      <td>1</td>
      <td>2006-02-14 22:04:36</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>5</th>
      <td>361</td>
      <td>2</td>
      <td>LAWRENCE</td>
      <td>LAWTON</td>
      <td>LAWRENCE.LAWTON@sakilacustomer.org</td>
      <td>366</td>
      <td>1</td>
      <td>2006-02-14 22:04:37</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>6</th>
      <td>421</td>
      <td>1</td>
      <td>LEE</td>
      <td>HAWKS</td>
      <td>LEE.HAWKS@sakilacustomer.org</td>
      <td>426</td>
      <td>1</td>
      <td>2006-02-14 22:04:37</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>7</th>
      <td>482</td>
      <td>1</td>
      <td>MAURICE</td>
      <td>CRAWLEY</td>
      <td>MAURICE.CRAWLEY@sakilacustomer.org</td>
      <td>487</td>
      <td>0</td>
      <td>2006-02-14 22:04:37</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
    <tr>
      <th>8</th>
      <td>499</td>
      <td>2</td>
      <td>MARC</td>
      <td>OUTLAW</td>
      <td>MARC.OUTLAW@sakilacustomer.org</td>
      <td>504</td>
      <td>1</td>
      <td>2006-02-14 22:04:37</td>
      <td>2006-02-15 04:57:20</td>
    </tr>
  </tbody>
</table>
</div>

```python

```
