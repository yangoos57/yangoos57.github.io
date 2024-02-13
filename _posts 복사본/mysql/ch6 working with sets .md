---
title: "ch6 working with sets "
category: "MySQL"
date: "2022-03-23"
thumbnail: "./mysql.png"
---

```python
import pandas as pd
import pymysql

conn = pymysql.connect(host='localhost', port=int(3306), user='root',passwd='1234', db='sakila')
cursor = conn.cursor(pymysql.cursors.DictCursor)
```

### The union Operator

- UNION (=A U B) : 중복 제거 함

- UNION ALL(= A + B) : 중복 제거 안함

```python
sen = '''

SELECT 'CUST' typ, c.first_name, c.last_name
FROM customer AS c
UNION ALL
SELECT 'ACTR' typ, a.first_name, a.last_name
FROM actor AS a;

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
      <th>typ</th>
      <th>first_name</th>
      <th>last_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CUST</td>
      <td>MARY</td>
      <td>SMITH</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CUST</td>
      <td>PATRICIA</td>
      <td>JOHNSON</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CUST</td>
      <td>LINDA</td>
      <td>WILLIAMS</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CUST</td>
      <td>BARBARA</td>
      <td>JONES</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CUST</td>
      <td>ELIZABETH</td>
      <td>BROWN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>794</th>
      <td>ACTR</td>
      <td>BELA</td>
      <td>WALKEN</td>
    </tr>
    <tr>
      <th>795</th>
      <td>ACTR</td>
      <td>REESE</td>
      <td>WEST</td>
    </tr>
    <tr>
      <th>796</th>
      <td>ACTR</td>
      <td>MARY</td>
      <td>KEITEL</td>
    </tr>
    <tr>
      <th>797</th>
      <td>ACTR</td>
      <td>JULIA</td>
      <td>FAWCETT</td>
    </tr>
    <tr>
      <th>798</th>
      <td>ACTR</td>
      <td>THORA</td>
      <td>TEMPLE</td>
    </tr>
  </tbody>
</table>
<p>799 rows × 3 columns</p>
</div>

### Set Operation Rules

compound Queries 사용시 주의해야 할 사항

1.  set을 쓰면 첫번째 set의 column을 사용하므로 order by는 첫번쨰 column 명으로 써야함.

```python
sen ='''

SELECT a.first_name fname, a.last_name lname
FROM actor a
WHERE a.first_name LIKE 'J%' AND a.last_name LIKE 'D%'
UNION ALL
SELECT c.first_name, c.last_name
FROM customer c
WHERE c.first_name LIKE 'J%' AND c.last_name LIKE 'D%'
ORDER BY lname,fname;

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
      <th>fname</th>
      <th>lname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JENNIFER</td>
      <td>DAVIS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JENNIFER</td>
      <td>DAVIS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JUDY</td>
      <td>DEAN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JODIE</td>
      <td>DEGENERES</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JULIANNE</td>
      <td>DENCH</td>
    </tr>
  </tbody>
</table>
</div>

2. set operator를 쓰는 위치에 따라서 결과 값이 달라질 수 있다.

```python

# UNION ALL -> UNION
sen ='''

SELECT a.first_name fname, a.last_name lname
FROM actor a
WHERE a.first_name LIKE 'J%' AND a.last_name LIKE 'D%'
UNION ALL
SELECT a.first_name, a.last_name
FROM actor a
WHERE a.first_name LIKE 'M%' AND a.last_name LIKE 'T%'
UNION
SELECT c.first_name, c.last_name
FROM customer c
WHERE c.first_name LIKE 'J%' AND c.last_name LIKE 'D%'
ORDER BY lname,fname;

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
      <th>fname</th>
      <th>lname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JENNIFER</td>
      <td>DAVIS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JUDY</td>
      <td>DEAN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JODIE</td>
      <td>DEGENERES</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JULIANNE</td>
      <td>DENCH</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MARY</td>
      <td>TANDY</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MENA</td>
      <td>TEMPLE</td>
    </tr>
  </tbody>
</table>
</div>

```python
# UNION -> UNION ALL
sen ='''

SELECT a.first_name fname, a.last_name lname
FROM actor a
WHERE a.first_name LIKE 'J%' AND a.last_name LIKE 'D%'
UNION
SELECT a.first_name, a.last_name
FROM actor a
WHERE a.first_name LIKE 'M%' AND a.last_name LIKE 'T%'
UNION ALL
SELECT c.first_name, c.last_name
FROM customer c
WHERE c.first_name LIKE 'J%' AND c.last_name LIKE 'D%'
ORDER BY lname,fname;

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
      <th>fname</th>
      <th>lname</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>JENNIFER</td>
      <td>DAVIS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JENNIFER</td>
      <td>DAVIS</td>
    </tr>
    <tr>
      <th>2</th>
      <td>JUDY</td>
      <td>DEAN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>JODIE</td>
      <td>DEGENERES</td>
    </tr>
    <tr>
      <th>4</th>
      <td>JULIANNE</td>
      <td>DENCH</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MARY</td>
      <td>TANDY</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MENA</td>
      <td>TEMPLE</td>
    </tr>
  </tbody>
</table>
</div>

### Exercise

**Exercise 6-1**

1. L M N O P Q R S T
2. L M N O P P Q R S T
3. P
4. L M N O

```python
### Exercise 6-2

sen = '''

SELECT a.first_name, a.last_name
FROM actor a
WHERE a.last_name LIKE 'L%'
UNION
SELECT c.first_name, c.last_name
FROM customer c
WHERE c.last_name LIKE 'L%'
ORDER BY last_name ;

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
      <td>MISTY</td>
      <td>LAMBERT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JACOB</td>
      <td>LANCE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RENEE</td>
      <td>LANE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HEIDI</td>
      <td>LARSON</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DARYL</td>
      <td>LARUE</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LAURIE</td>
      <td>LAWRENCE</td>
    </tr>
    <tr>
      <th>6</th>
      <td>JEANNE</td>
      <td>LAWSON</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LAWRENCE</td>
      <td>LAWTON</td>
    </tr>
    <tr>
      <th>8</th>
      <td>KIMBERLY</td>
      <td>LEE</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MATTHEW</td>
      <td>LEIGH</td>
    </tr>
    <tr>
      <th>10</th>
      <td>LOUIS</td>
      <td>LEONE</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SARAH</td>
      <td>LEWIS</td>
    </tr>
    <tr>
      <th>12</th>
      <td>GEORGE</td>
      <td>LINTON</td>
    </tr>
    <tr>
      <th>13</th>
      <td>MAUREEN</td>
      <td>LITTLE</td>
    </tr>
    <tr>
      <th>14</th>
      <td>JOHNNY</td>
      <td>LOLLOBRIGIDA</td>
    </tr>
    <tr>
      <th>15</th>
      <td>DWIGHT</td>
      <td>LOMBARDI</td>
    </tr>
    <tr>
      <th>16</th>
      <td>JACQUELINE</td>
      <td>LONG</td>
    </tr>
    <tr>
      <th>17</th>
      <td>AMY</td>
      <td>LOPEZ</td>
    </tr>
    <tr>
      <th>18</th>
      <td>BARRY</td>
      <td>LOVELACE</td>
    </tr>
    <tr>
      <th>19</th>
      <td>PRISCILLA</td>
      <td>LOWE</td>
    </tr>
    <tr>
      <th>20</th>
      <td>VELMA</td>
      <td>LUCAS</td>
    </tr>
    <tr>
      <th>21</th>
      <td>WILLARD</td>
      <td>LUMPKIN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>LEWIS</td>
      <td>LYMAN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>JACKIE</td>
      <td>LYNCH</td>
    </tr>
  </tbody>
</table>
</div>

```python

```
