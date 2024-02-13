---
title: "9. cleaning up the table"
category: "Datapreporcessing"
date: "2022-05-09"
thumbnail: "./data/preprocessing.png"
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.getcwd()

print(pd.__version__)

```

내가 원하는 자료를 찾기도 어렵구나..

나중에 분석연습을 할때는 계획 후 행동하는 연습도 같이해보자.

데이터 전처리를 할 땐 우리가 어떤 목표로 이런 작업을 하는지 이해해야합니다. 특히 우리가 지금 데이터로 무엇을 구현하고자 하는지가 중요합니다.

분석 방법에 따라 필요한 데이터 형태가 다르기 때문이에요. 그래서 어떤 분석도구를 사용하느냐에 따라서 데이터 전처리 방식도 달라집니다.

```python
filename = os.listdir("./data/ch9/speeches")
print(filename)
```

    ['BattleCreekDec19_2019.txt', 'BemidjiSep18_2020.txt', 'CharlestonFeb28_2020.txt', 'CharlotteMar2_2020.txt', 'CincinnatiAug1_2019.txt', 'ColoradorSpringsFeb20_2020.txt', 'DallasOct17_2019.txt', 'DesMoinesJan30_2020.txt', 'FayettevilleSep19_2020.txt', 'FayettevilleSep9_2019.txt', 'FreelandSep10_2020.txt', 'GreenvilleJul17_2019.txt', 'HendersonSep13_2020.txt', 'HersheyDec10_2019.txt', 'LasVegasFeb21_2020.txt', 'LatrobeSep3_2020.txt', 'LexingtonNov4_2019.txt', 'MilwaukeeJan14_2020.txt', 'MindenSep12_2020.txt', 'MinneapolisOct10_2019.txt', 'MosineeSep17_2020.txt', 'NewHampshireAug15_2019.txt', 'NewHampshireAug28_2020.txt', 'NewHampshireFeb10_2020.txt', 'NewMexicoSep16_2019.txt', 'OhioSep21_2020.txt', 'PhoenixFeb19_2020.txt', 'PittsburghSep22_2020.txt', 'TexasSep23_2019.txt', 'ToledoJan9_2020.txt', 'TulsaJun20_2020.txt', 'TupeloNov1_2019.txt', 'WildwoodJan28_2020.txt', 'Winston-SalemSep8_2020.txt', 'YumaAug18_2020.txt']

```python
speech_df = pd.DataFrame(index=range(len(filename)), columns=['file name', 'the content'])
```

- 파일 불러오기 : open(경로)
- 텍스트(Text) 읽기 : read(), readline(), readlines()
- 폴더 내 파일 목록 불러오기 : os.listdir()

```python
for i, f_name in enumerate(filename) :
    f = open('data/ch9/speeches/' + f_name, "r", encoding='utf-8')
    f_content = f.readlines()
    f.close()
    speech_df.at[i,'file name'] = f_name
    speech_df.at[i, 'the content'] = f_content[0]

speech_df.columns = ['names', 'contents']

speech_df.head(1)
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
      <th>names</th>
      <th>contents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BattleCreekDec19_2019.txt</td>
      <td>Thank you. Thank you. Thank you to Vice Presid...</td>
    </tr>
  </tbody>
</table>
</div>

```python
air_df = pd.read_csv('./data/ch9/TempData.csv')
air_df
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
      <th>Temp</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>79.0</td>
      <td>2016</td>
      <td>1</td>
      <td>1</td>
      <td>00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>79.0</td>
      <td>2016</td>
      <td>1</td>
      <td>1</td>
      <td>00:30:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>79.0</td>
      <td>2016</td>
      <td>1</td>
      <td>1</td>
      <td>01:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77.0</td>
      <td>2016</td>
      <td>1</td>
      <td>1</td>
      <td>01:30:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>78.0</td>
      <td>2016</td>
      <td>1</td>
      <td>1</td>
      <td>02:00:00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20448</th>
      <td>77.0</td>
      <td>2016</td>
      <td>12</td>
      <td>31</td>
      <td>22:00:00</td>
    </tr>
    <tr>
      <th>20449</th>
      <td>77.0</td>
      <td>2016</td>
      <td>12</td>
      <td>31</td>
      <td>22:30:00</td>
    </tr>
    <tr>
      <th>20450</th>
      <td>77.0</td>
      <td>2016</td>
      <td>12</td>
      <td>31</td>
      <td>23:00:00</td>
    </tr>
    <tr>
      <th>20451</th>
      <td>77.0</td>
      <td>2016</td>
      <td>12</td>
      <td>31</td>
      <td>23:00:00</td>
    </tr>
    <tr>
      <th>20452</th>
      <td>77.0</td>
      <td>2016</td>
      <td>12</td>
      <td>31</td>
      <td>23:30:00</td>
    </tr>
  </tbody>
</table>
<p>20453 rows × 5 columns</p>
</div>

멀티 인덱스 설정하는 방법

```python
air_df.drop(columns =['Year'], inplace=True)
air_df.set_index(['Month', 'Day', 'Time'], inplace=True)
```

```python
air_df
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
      <th></th>
      <th></th>
      <th>Temp</th>
    </tr>
    <tr>
      <th>Month</th>
      <th>Day</th>
      <th>Time</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">1</th>
      <th rowspan="5" valign="top">1</th>
      <th>00:00:00</th>
      <td>79.0</td>
    </tr>
    <tr>
      <th>00:30:00</th>
      <td>79.0</td>
    </tr>
    <tr>
      <th>01:00:00</th>
      <td>79.0</td>
    </tr>
    <tr>
      <th>01:30:00</th>
      <td>77.0</td>
    </tr>
    <tr>
      <th>02:00:00</th>
      <td>78.0</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">12</th>
      <th rowspan="5" valign="top">31</th>
      <th>22:00:00</th>
      <td>77.0</td>
    </tr>
    <tr>
      <th>22:30:00</th>
      <td>77.0</td>
    </tr>
    <tr>
      <th>23:00:00</th>
      <td>77.0</td>
    </tr>
    <tr>
      <th>23:00:00</th>
      <td>77.0</td>
    </tr>
    <tr>
      <th>23:30:00</th>
      <td>77.0</td>
    </tr>
  </tbody>
</table>
<p>20453 rows × 1 columns</p>
</div>

```python
response_df = pd.read_csv('./data/ch9/OSMI Mental Health in Tech Survey 2019.csv')
response_df
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
      <th>*Are you self-employed?*</th>
      <th>How many employees does your company or organization have?</th>
      <th>Is your employer primarily a tech company/organization?</th>
      <th>Is your primary role within your company related to tech/IT?</th>
      <th>Does your employer provide mental health benefits as part of healthcare coverage?</th>
      <th>Do you know the options for mental health care available under your employer-provided health coverage?</th>
      <th>Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?</th>
      <th>Does your employer offer resources to learn more about mental health disorders and options for seeking help?</th>
      <th>Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?</th>
      <th>If a mental health issue prompted you to request a medical leave from work, how easy or difficult would it be to ask for that leave?</th>
      <th>...</th>
      <th>Briefly describe what you think the industry as a whole and/or employers could do to improve mental health support for employees.</th>
      <th>If there is anything else you would like to tell us that has not been covered by the survey questions, please use this space to do so.</th>
      <th>Would you be willing to talk to one of us more extensively about your experiences with mental health issues in the tech industry? (Note that all interview responses would be used _anonymously_ and only with your permission.)</th>
      <th>What is your age?</th>
      <th>What is your gender?</th>
      <th>What country do you *live* in?</th>
      <th>What US state or territory do you *live* in?</th>
      <th>What is your race?</th>
      <th>What country do you *work* in?</th>
      <th>What US state or territory do you *work* in?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>26-100</td>
      <td>True</td>
      <td>True</td>
      <td>I don't know</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>I don't know</td>
      <td>Very easy</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>25</td>
      <td>Male</td>
      <td>United States of America</td>
      <td>Nebraska</td>
      <td>White</td>
      <td>United States of America</td>
      <td>Nebraska</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>26-100</td>
      <td>True</td>
      <td>True</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>I don't know</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>51</td>
      <td>male</td>
      <td>United States of America</td>
      <td>Nebraska</td>
      <td>White</td>
      <td>United States of America</td>
      <td>Nebraska</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>26-100</td>
      <td>True</td>
      <td>True</td>
      <td>I don't know</td>
      <td>No</td>
      <td>No</td>
      <td>I don't know</td>
      <td>I don't know</td>
      <td>Somewhat difficult</td>
      <td>...</td>
      <td>I think opening up more conversation around th...</td>
      <td>Thank you</td>
      <td>True</td>
      <td>27</td>
      <td>Male</td>
      <td>United States of America</td>
      <td>Illinois</td>
      <td>White</td>
      <td>United States of America</td>
      <td>Illinois</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>100-500</td>
      <td>True</td>
      <td>True</td>
      <td>I don't know</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Very easy</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>37</td>
      <td>male</td>
      <td>United States of America</td>
      <td>Nebraska</td>
      <td>White</td>
      <td>United States of America</td>
      <td>Nebraska</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>26-100</td>
      <td>True</td>
      <td>True</td>
      <td>I don't know</td>
      <td>No</td>
      <td>I don't know</td>
      <td>I don't know</td>
      <td>I don't know</td>
      <td>I don't know</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>46</td>
      <td>m</td>
      <td>United States of America</td>
      <td>Nebraska</td>
      <td>White</td>
      <td>United States of America</td>
      <td>Nebraska</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <th>347</th>
      <td>False</td>
      <td>More than 1000</td>
      <td>False</td>
      <td>True</td>
      <td>I don't know</td>
      <td>No</td>
      <td>No</td>
      <td>I don't know</td>
      <td>I don't know</td>
      <td>Somewhat difficult</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>27</td>
      <td>male</td>
      <td>India</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>India</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>348</th>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>IDK</td>
      <td>NaN</td>
      <td>False</td>
      <td>48</td>
      <td>m</td>
      <td>United States of America</td>
      <td>Louisiana</td>
      <td>White</td>
      <td>United States of America</td>
      <td>Louisiana</td>
    </tr>
    <tr>
      <th>349</th>
      <td>True</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>sdsdsdsdsdsd</td>
      <td>sdsdsdsdsdsd</td>
      <td>False</td>
      <td>50</td>
      <td>M</td>
      <td>India</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>India</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>350</th>
      <td>False</td>
      <td>More than 1000</td>
      <td>True</td>
      <td>True</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Difficult</td>
      <td>...</td>
      <td>raise awareness</td>
      <td>no</td>
      <td>False</td>
      <td>30</td>
      <td>female</td>
      <td>India</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>India</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>351</th>
      <td>False</td>
      <td>More than 1000</td>
      <td>True</td>
      <td>True</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Somewhat difficult</td>
      <td>...</td>
      <td>reduce stigma. offer options for part time wor...</td>
      <td>I've had to drive all of the progress in menta...</td>
      <td>True</td>
      <td>24</td>
      <td>Female (cis)</td>
      <td>United States of America</td>
      <td>Oregon</td>
      <td>White</td>
      <td>United States of America</td>
      <td>Oregon</td>
    </tr>
  </tbody>
</table>
<p>352 rows × 82 columns</p>
</div>

```python
new_col = [f'Q{i}' for i in range(1,83)]
k = response_df.columns.to_list()
columns_dict = pd.Series(k,index = new_col)

```

```python
keys = ['Q{}'.format(i) for i in range(1,83)]
columns_dic = pd.Series(response_df.columns,index=keys)
columns_dic
```

    Q1                              *Are you self-employed?*
    Q2     How many employees does your company or organi...
    Q3     Is your employer primarily a tech company/orga...
    Q4     Is your primary role within your company relat...
    Q5     Does your employer provide mental health benef...
                                 ...
    Q78                       What country do you *live* in?
    Q79         What US state or territory do you *live* in?
    Q80                                   What is your race?
    Q81                       What country do you *work* in?
    Q82         What US state or territory do you *work* in?
    Length: 82, dtype: object

### Exercise 1

```python
file_list =os.listdir("./data/ch9/SBID_Data")

```

```python
file_list =os.listdir("./data/ch9/SBID_Data")
title = "./data/ch9/SBID_Data/"+file_list[0]
test = pd.read_excel(title)
test.head(10)

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
      <th>Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00:00:30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00:01:18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00:01:19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00:01:28</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00:01:34</td>
    </tr>
    <tr>
      <th>5</th>
      <td>00:01:54</td>
    </tr>
    <tr>
      <th>6</th>
      <td>00:02:11</td>
    </tr>
    <tr>
      <th>7</th>
      <td>00:02:20</td>
    </tr>
    <tr>
      <th>8</th>
      <td>00:02:24</td>
    </tr>
    <tr>
      <th>9</th>
      <td>00:04:11</td>
    </tr>
  </tbody>
</table>
</div>

### Datetime.date => Datetime64로 바꾸기

### pd.date_range() => 시간 range 만들기

### pd. => 시간 단위로 row 개수 세기

https://stackoverflow.com/questions/47362530/python-pandas-group-datetimes-by-hour-and-count-row

```python
file_list =os.listdir("./data/ch9/SBID_Data")
```

```python
# 1. 파일 불러오기
# 2. 시간단위로 row 개수 세는 series 만들기
test_con =[]
for i in range(len(file_list)) :
    title = "./data/ch9/SBID_Data/"+file_list[i]
    test = pd.read_excel(title)
    test.Time = pd.to_datetime(test.Time.astype(str)) # datetime.time => datetime64로 만들기
    test_count = test.groupby([pd.Grouper(key='Time',freq='H')]).size()
    test_count.index=[k for k in range(1,25)]
    test_con.append(test_count)

```

```python
len(test_con)
```

    14

```python
# 3. concat으로 multiindex 만들기
test_keys = range(20201012,20201026)
test_total = pd.concat(test_con, keys=test_keys)
test_total
```

    20201012  1      60
              2      73
              3      63
              4      70
              5      72
                   ...
    20201025  20    616
              21     65
              22     59
              23     54
              24     75
    Length: 336, dtype: int64

```python
from ipywidgets import interact, widgets

# concat 한 파일을 불러오기
def test_interact(test_date) :
    plt.figure(figsize=(10,8))
    test_count=test_total.loc[test_date]
    plt.bar(test_count.index,test_count.values,color='orange')
    for i in range(24) :
        row = test_count[i+1]
        plt.annotate(row, (i+0.5, row+10))
    plt.xlim(0.5,24.5)
    plt.ylim(0,max(test_count.values)+100)

interact(test_interact, test_date = widgets.IntSlider(min=20201012, max=20201025, step=1, value=20201012))
```

    interactive(children=(IntSlider(value=20201012, description='test_date', max=20201025, min=20201012), Output()…





    <function __main__.test_interact(test_date)>
