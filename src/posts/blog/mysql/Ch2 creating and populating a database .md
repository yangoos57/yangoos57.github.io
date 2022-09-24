---
title: "Ch2 creating and populating a database"
category: "MySQL"
date: "2022-03-19"
thumbnail: "./mysql.png"
---

```python
import pandas as pd
import pymysql
conn=pymysql.connect(host='localhost',port=int(3306),user='root',passwd='1234',db='learning_sql')
cursor = conn.cursor(pymysql.cursors.DictCursor)
```

### 연습용 Data 만들기

- desc : describe의 약자인듯

```python
cursor.execute(' desc person')
result = cursor.fetchall()
pd.DataFrame(result)
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
      <td>person_id</td>
      <td>smallint unsigned</td>
      <td>NO</td>
      <td>PRI</td>
      <td>None</td>
      <td>auto_increment</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fname</td>
      <td>varchar(20)</td>
      <td>YES</td>
      <td></td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>lname</td>
      <td>varchar(20)</td>
      <td>YES</td>
      <td></td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>eye_color</td>
      <td>enum('BR','BL','GR')</td>
      <td>YES</td>
      <td></td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>birth_date</td>
      <td>date</td>
      <td>YES</td>
      <td></td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>5</th>
      <td>street</td>
      <td>varchar(30)</td>
      <td>YES</td>
      <td></td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>6</th>
      <td>city</td>
      <td>varchar(20)</td>
      <td>YES</td>
      <td></td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>7</th>
      <td>state</td>
      <td>varchar(20)</td>
      <td>YES</td>
      <td></td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>8</th>
      <td>country</td>
      <td>varchar(20)</td>
      <td>YES</td>
      <td></td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>9</th>
      <td>postal_code</td>
      <td>varchar(20)</td>
      <td>YES</td>
      <td></td>
      <td>None</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>

```python
sen = '''

DESC favorite_food

'''
cursor.execute(sen)
result = cursor.fetchall()
pd.DataFrame(result)
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
      <td>person_id</td>
      <td>smallint unsigned</td>
      <td>NO</td>
      <td>PRI</td>
      <td>None</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>food</td>
      <td>varchar(20)</td>
      <td>NO</td>
      <td>PRI</td>
      <td>None</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>

### populating and modifying tables

- insert statement로 row를 추가한다.

- insert에 들어가지 않았거나 수정하려면 update statement를 쓴다.

```python
sen ='''SELECT * FROM person'''

cursor.execute(sen)
result = cursor.fetchall()
pd.DataFrame(result)
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
      <th>person_id</th>
      <th>fname</th>
      <th>lname</th>
      <th>eye_color</th>
      <th>birth_date</th>
      <th>street</th>
      <th>city</th>
      <th>state</th>
      <th>country</th>
      <th>postal_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>William</td>
      <td>Turner</td>
      <td>BR</td>
      <td>1972-05-27</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>

```python
sen = '''
INSERT INTO person
(person_id, fname, lname, eye_color, birth_date)
VALUES (null, 'William','Turner', 'BR', '1972-05-27');
'''

cursor.execute(sen)
result = cursor.fetchall()
conn.commit()
```

```python
sen = '''
SELECT *
FROM person
'''

cursor.execute(sen)
result = cursor.fetchall()
pd.DataFrame(result)
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
      <th>person_id</th>
      <th>fname</th>
      <th>lname</th>
      <th>eye_color</th>
      <th>birth_date</th>
      <th>street</th>
      <th>city</th>
      <th>state</th>
      <th>country</th>
      <th>postal_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>William</td>
      <td>Turner</td>
      <td>BR</td>
      <td>1972-05-27</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>William</td>
      <td>Turner</td>
      <td>BR</td>
      <td>1972-05-27</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>

### Table에서 원하는 column 불러오기

- SELECT columns*명 FROM table*명

```python
sen = '''
SELECT *
FROM person
'''

cursor.execute(sen)
result = cursor.fetchall()
pd.DataFrame(result)
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
      <th>person_id</th>
      <th>fname</th>
      <th>lname</th>
      <th>eye_color</th>
      <th>birth_date</th>
      <th>street</th>
      <th>city</th>
      <th>state</th>
      <th>country</th>
      <th>postal_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>William</td>
      <td>Turner</td>
      <td>BR</td>
      <td>1972-05-27</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>William</td>
      <td>Turner</td>
      <td>BR</td>
      <td>1972-05-27</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>

### 필터걸기

- Where 사용하면 필터를 걸 수 있다.

```python
sen = '''
SELECT person_id, fname, lname, birth_date
FROM person
WHERE person_id = 6
'''

cursor.execute(sen)
result = cursor.fetchall()
pd.DataFrame(result)
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
      <th>person_id</th>
      <th>fname</th>
      <th>lname</th>
      <th>birth_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>William</td>
      <td>Turner</td>
      <td>1972-05-27</td>
    </tr>
  </tbody>
</table>
</div>

### ROW 추가하기

- INSERT INTO table (column) values (values)

```python
data = [(6, 'pizza'),(6, 'cookies'),(6, 'nachos')]

sen = '''
INSERT INTO favorite_food (person_id, food) VALUES (%s, %s);
'''
cursor.executemany(sen,data)
result = cursor.fetchall()
pd.DataFrame(result)
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
SELECT food
FROM favorite_food
WHERE person_id = 6
ORDER BY food;
'''

cursor.execute(sen)
result = cursor.fetchall()
pd.DataFrame(result)
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
      <th>food</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cookies</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nachos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pizza</td>
    </tr>
  </tbody>
</table>
</div>

```python
sen = '''
INSERT INTO person
(person_id, fname, lname, eye_color, birth_date,
street, city, state, country, postal_code)
VALUES (null, 'Susan','Smith', 'BL', '1975-11-02',
'23 Maple St.', 'Arlington', 'VA', 'USA', '20220');
'''

cursor.execute(sen)
result = cursor.fetchall()
pd.DataFrame(result)
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
SELECT person_id, fname, lname, birth_date
FROM person
'''

cursor.execute(sen)
result = cursor.fetchall()
pd.DataFrame(result)
conn.commit()
```

```python
sen = '''
UPDATE person
SET street = '1',
city='boston',
country = 'USA',
postal_code = '02138'
WHERE person_id = 6
'''

cursor.execute(sen)
conn.commit()
result = cursor.fetchall()
pd.DataFrame(result)
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
