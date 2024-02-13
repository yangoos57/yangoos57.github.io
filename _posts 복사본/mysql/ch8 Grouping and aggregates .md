---
title: "ch8 Grouping and aggregates"
category: "MySQL"
date: "2022-03-24"
thumbnail: "./mysql.png"
---

```python
import pandas as pd
import pymysql

conn = pymysql.connect(host='localhost', port=int(3306), user='root',passwd='1234', db='sakila')
cursor = conn.cursor(pymysql.cursors.DictCursor)
```

### Aggregate Functions

max()

min()

avg()

sum()

count()

> Aggregate Function을 사용할 땐 WHERE을 쓰면 오류가 난다.
>
> WHERE우선 적용된뒤 GROUP BY가 이뤄지기 때문이다.
>
> 따라서 `HAVING`을 사용해야한다.

```python
sen = '''

SELECT customer_id, count(*)
FROM rental
GROUP BY customer_id
HAVING count(*) >= 40
ORDER BY 2 DESC;
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
      <th>count(*)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>148</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1</th>
      <td>526</td>
      <td>45</td>
    </tr>
    <tr>
      <th>2</th>
      <td>144</td>
      <td>42</td>
    </tr>
    <tr>
      <th>3</th>
      <td>236</td>
      <td>42</td>
    </tr>
    <tr>
      <th>4</th>
      <td>75</td>
      <td>41</td>
    </tr>
    <tr>
      <th>5</th>
      <td>197</td>
      <td>40</td>
    </tr>
    <tr>
      <th>6</th>
      <td>469</td>
      <td>40</td>
    </tr>
  </tbody>
</table>
</div>

```python
sen = '''

SELECT MAX(amount) max_amt, MIN(amount) min_amt, AVG(amount) avg_amt, SUM(amount) tot_amt, COUNT(*) num_payments
FROM payment;

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
      <th>max_amt</th>
      <th>min_amt</th>
      <th>avg_amt</th>
      <th>tot_amt</th>
      <th>num_payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>11.99</td>
      <td>0.00</td>
      <td>4.200667</td>
      <td>67416.51</td>
      <td>16049</td>
    </tr>
  </tbody>
</table>
</div>

```python
sen = '''

SELECT customer_id, MAX(amount) max_amt, MIN(amount) min_amt, AVG(amount) avg_amt, SUM(amount) tot_amt, COUNT(*) num_payments
FROM payment
GROUP BY customer_id;
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
      <th>max_amt</th>
      <th>min_amt</th>
      <th>avg_amt</th>
      <th>tot_amt</th>
      <th>num_payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>9.99</td>
      <td>0.99</td>
      <td>3.708750</td>
      <td>118.68</td>
      <td>32</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>10.99</td>
      <td>0.99</td>
      <td>4.767778</td>
      <td>128.73</td>
      <td>27</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>10.99</td>
      <td>0.99</td>
      <td>5.220769</td>
      <td>135.74</td>
      <td>26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>8.99</td>
      <td>0.99</td>
      <td>3.717273</td>
      <td>81.78</td>
      <td>22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>9.99</td>
      <td>0.99</td>
      <td>3.805789</td>
      <td>144.62</td>
      <td>38</td>
    </tr>
    <tr>
      <th>...</th>
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
      <td>10.99</td>
      <td>0.99</td>
      <td>3.923333</td>
      <td>117.70</td>
      <td>30</td>
    </tr>
    <tr>
      <th>595</th>
      <td>596</td>
      <td>6.99</td>
      <td>0.99</td>
      <td>3.454286</td>
      <td>96.72</td>
      <td>28</td>
    </tr>
    <tr>
      <th>596</th>
      <td>597</td>
      <td>8.99</td>
      <td>0.99</td>
      <td>3.990000</td>
      <td>99.75</td>
      <td>25</td>
    </tr>
    <tr>
      <th>597</th>
      <td>598</td>
      <td>7.99</td>
      <td>0.99</td>
      <td>3.808182</td>
      <td>83.78</td>
      <td>22</td>
    </tr>
    <tr>
      <th>598</th>
      <td>599</td>
      <td>9.99</td>
      <td>0.99</td>
      <td>4.411053</td>
      <td>83.81</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
<p>599 rows × 6 columns</p>
</div>

```python
sen = '''

SELECT COUNT(customer_id) num_rows, COUNT(DISTINCT customer_id) num_customers
FROM payment;

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
      <th>num_rows</th>
      <th>num_customers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16049</td>
      <td>599</td>
    </tr>
  </tbody>
</table>
</div>

### datediff

날짜를 반환함

```python
sen = '''

SELECT MAX(datediff(return_date, rental_date)) coffee
FROM rental ;
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
      <th>coffee</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>

### Multicolumn Grouping

```python
sen = '''

SELECT fa.actor_id, f.rating, count(*)
FROM film_actor fa
    INNER JOIN film f
    ON fa.film_id = f.film_id
GROUP BY fa.actor_id, f.rating
ORDER BY 1,2;

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
      <th>rating</th>
      <th>count(*)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>G</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>PG</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>PG-13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>R</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>NC-17</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>991</th>
      <td>200</td>
      <td>G</td>
      <td>5</td>
    </tr>
    <tr>
      <th>992</th>
      <td>200</td>
      <td>PG</td>
      <td>3</td>
    </tr>
    <tr>
      <th>993</th>
      <td>200</td>
      <td>PG-13</td>
      <td>2</td>
    </tr>
    <tr>
      <th>994</th>
      <td>200</td>
      <td>R</td>
      <td>6</td>
    </tr>
    <tr>
      <th>995</th>
      <td>200</td>
      <td>NC-17</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>996 rows × 3 columns</p>
</div>

```python
sen = '''



'''

cursor.execute(sen)
pd.DataFrame(cursor.fetchall())
```

```python
sen = '''

SELECT extract(YEAR FROM rental_date) year,
    COUNT(*) how_many
FROM rental
GROUP BY year;


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
      <th>year</th>
      <th>how_many</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2005</td>
      <td>15862</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2006</td>
      <td>182</td>
    </tr>
  </tbody>
</table>
</div>

### ROLL UP

Group by 할 때 total을 새로 만듬

actor_id 1에 대한 rating 전체의 합 = WITH ROLLUP

```python
sen = '''

SELECT fa.actor_id, f.rating, count(*)
FROM film_actor fa
    INNER JOIN film f
    ON fa.film_id = f.film_id
GROUP BY fa.actor_id, f.rating WITH ROLLUP
ORDER BY 1,2;

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
      <th>rating</th>
      <th>count(*)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>None</td>
      <td>5462</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>None</td>
      <td>19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>G</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>NC-17</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>PG</td>
      <td>6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1192</th>
      <td>200.0</td>
      <td>G</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1193</th>
      <td>200.0</td>
      <td>NC-17</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1194</th>
      <td>200.0</td>
      <td>PG</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1195</th>
      <td>200.0</td>
      <td>PG-13</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>200.0</td>
      <td>R</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>1197 rows × 3 columns</p>
</div>

### Having 복습

When adding filters to a query that includes a group by clause,

think carefully about whether the filter acts on raw data, in which

case it belongs in the where clause, or on grouped data, in which

case it belongs in the having clause.

```python
sen = '''

SELECT fa.actor_id, f.rating, count(*)
FROM film_actor fa
    INNER JOIN film f
    ON fa.film_id = f.film_id
GROUP BY fa.actor_id, f.rating
HAVING count(*) > 9
ORDER BY 2;

'''

cursor.execute(sen)
# pd.DataFrame(cursor.fetchall())
```

    48

```python
# Exercise 8-1,2,3

sen = '''

SELECT customer_id, count(*), sum(amount)
FROM payment
GROUP BY customer_id
HAVING count(*) >= 40 ;
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
      <th>count(*)</th>
      <th>sum(amount)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75</td>
      <td>41</td>
      <td>155.59</td>
    </tr>
    <tr>
      <th>1</th>
      <td>144</td>
      <td>42</td>
      <td>195.58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>148</td>
      <td>46</td>
      <td>216.54</td>
    </tr>
    <tr>
      <th>3</th>
      <td>197</td>
      <td>40</td>
      <td>154.60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>236</td>
      <td>42</td>
      <td>175.58</td>
    </tr>
    <tr>
      <th>5</th>
      <td>469</td>
      <td>40</td>
      <td>177.60</td>
    </tr>
    <tr>
      <th>6</th>
      <td>526</td>
      <td>45</td>
      <td>221.55</td>
    </tr>
  </tbody>
</table>
</div>
