# Python

Content based strongly on the Data Analysis course: https://www.freecodecamp.org/

pip install pandas matplotlib scipy

import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

sales = pd.read_csv(
    'data/sales_data.csv',
		header=None,
		names=['', ...]
    parse_dates=['Date']
)

df = pd.read_sql(query, conn, index_col='', parse_dates=['', ''])

df = pd.DataFrame(data['Result']['3600'], columns=['',...])
df['column'] = pd.to_datetime(df['...'], units='s')

sales.head()
sales.shape
sales.info()

sales.describe()
sales['column'].describe()

sales['column']
	.mean/median.../max()
	.plot(kind='box/hist/pie/bar', vert=False, figsize=(14,6) bins=100)
	.plot(kind='scatter', x='', y='', figsize=(6,6))
	.value_counts()

ax = sales['column'].plot(kind='density', figsize=(14, 6))
ax.axline(sales['column'].mean(), color='red')
ax.set_ylabel('').set_xlabel('')

sales[['', '']].boxplot(by='', figsize=(10, 6))
sales[boxplot_cols].plot(kind='box', subplots=True, layout=(2,3), figsize=(14,8))

sales['new_column'] = ...
(sales['column1'] != sales['column2']).sum()

sales.loc[sales['column'] == 'value']
	.mean()

sales.loc[() &/| ()]
	.shape[0]
	.mean()

sales.loc[sales['column'] == sales['column'].max(), 'column_2'].unique()
sales.sort_value(['Revenue'], ascending=False).head(5)

sales['column'].isin(['val1', 'val2', ...])

writer = pd.ExcelWriter('...')
sales.to_excel(writer, sheet_name='...')

writer.save()


import numpy as np

x = 5 // 20 bytes

np.int8/16/32/64...
a = np.array([...], dtype=np.float)
a[0], a[0, 1], a[0, -1]

b[0], b[2], b[-1] == b[[0, 2, -1]] //other numpy array

a[1:-1] = b
b = a.copy()

a.dtype

a = np.array([
	[...],
	[...],
	...
])

a.shape
a.ndim
a.size

// warning: match the dimentions!

// Vectorized operations

a = a.arange(4)
a += ..., *=, /=, -=
a + b, -, *, /

// Boolean selection
a[[true, false, ...]]

// Condition selection
a <= 3, >=, !=, ==, <, >
a[(a < 4), ~, |, &]

// random arr with numbers from 0..99 with shape = 3x3
np.random.randint(100, size=(3, 3))

//one to one product
A*B
// matriz product
np.matmul(a,b)
// determinant
np.linalg.det(a)
// dot product
A.dot(B)
// cross product
A @ B
// transpose
B.T

np.vstack([...])
np.hstack([...])

sys.getsizeof(1) # 28
sys.getsizeof(10**100) # 72 (long)

%time sum(...)

np.random.random(size=2)
np.random.normal(size=2)
np.random.rand(2, 4)

np.arange(10)
np.arange(5, 10)
np.arange(0, 1, .1)

np.arange(10).reshape(2, 5)
np.arange(10).reshape(5, 2)

np.linspace(0, 1, 5)
np.linspace(0, 1, 20)
np.linspace(0, 1, 20, false)

np.zeros(5)
np.zeros((3, 3))
np.zeros((3,3), dtype=np.int)
np.ones((3, 3))
np.empty(5)
np.empty((2, 2))

np.identity(3)
np.eye(3, 3)
np.eye(8, 4)
np.eye(8, 4, k=1)
np.eye(8, 4, k=-3)
"hello World"[6]

np.ones_like(X)
np.random.randn(3, 3, 3) // 3x3x3 arr with float randoms [0, 1)
np.array(X) // convert pyArr to npArr
np.copy(X)
a.all()
a.any()

np.sum(axis=0)/mean()/max()

axis=0 columns
axis=1 rows


import pandas as pd
import numpy as np

// ordered dicc

sr = pd.Series([...], name='')
sr = pd.Series({... : ...}, name='')
sr.name = ...
sr.values
sr[0]
sr.index
sr.index = ...
sr.iloc[0]
sr[['', '', ...]]
sr.iloc[[0, 1]]
sr['':''] // slice includes last element

sr > ... // > >= < <= == !=
sr[sr > ...]

np.log(sr)


import numpy as np
import panda  as pd

df = pd.DataFrame({
	'Column': [...],
	...
})

// Rows
df.index = [
	...
]

df.min()/max()/sum()/mean()/std()/median()/quantile(.25)/quantile([.2, .4, ...])
df.head()
df.tail()
df.columns
df.index
df.info()
df.size
df.shape
df.describe()
df.dtypes
df.dtypes.value_counts()

df['column']/.to_frame()
df.loc['index/row']
df.iloc[1]

df.loc[[..., ...]]
df.loc[...:..., ...]

df[...] > ...
df.loc[df[...] > ...]

df = df.drop('')
df.drop([..., ....])
df.drop(columns=[...], axis=0/columns)

we can perform serie + dataframe matching indexes

 df.rename(
	columns={
		old:new,
	},
	index={...}
)

df.rename(columns=str.upper)
df.rename(columns=lambda x: x.lower())

df.append(pd.Series({
	..., ...
}))

df.reset_index()
df.set_index('Population')

df[]=df[] / df[]


pd.read_csv('...', header=None)
df[...] = pd.to_datetime(df[...])
df.set_index('', inplace=True)

pd.read_csv(
	'...',
	header=None,
	names=[..., ...],
	index_col=0,
	parse_dates=True
)


No sense values

df[...].unique()
df[...].value_counts()
df[...].replace('', '')

Duplicates

// Duplicated are all values except the first
df.duplicated(kepp='last'/False, subset=['column'])

df['columns'].str.split('-')/.replace() 
						 .dt
						 .cat


sales = pd.read_csv(
    'data/sales_data.csv',
    parse_dates=['Date']
)

corr = sales.corr()

// Global API, MATLAB

x = np.arange(-10, 11)
plt.figure(figsize=(12,6))
plt.title('...')

plt.subplot(1 ,2, 1) // rows, columns, selected

plt.plot(x, x**2)
plt.plot([0,0,0], [-10,0,100])
plt.legend([''])
plt.xlabel('...')
plt.ylabel('...')

plt.subplot(1, 2, 2)
...
plt.plot(x, -1 * (x**2))
...

// OOP interface

fig, axes = plt.subplots(figsize=(12,6))
axes.plot(
	x, x**2, color='red', linewidth=3,
	marker='o', markersize=8, label='X^2',
	linestyle='solid/dashed/dashdot/dotted green/red...'
)
axes.set_xlabel('X')
axes.set_ylabel('...')

fig <- to see the result

 // Other example (colors and points)
plt.figure(figsize=(14,6))

ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)

x = np.random.rand(N)
y = 
colors = 
area = np.pi * (20 * np.random.rand(N)) ** 2
plt.scatter(x, y, s=area, c=colors, alpha=0.5, cmap='Spectral')
plt.colorbar()
plt.show()

//Other example (hist)

plt.subplots(figsize=(12,6))
plt.hist(
	values, bins=100, alpha=0.8,
	histtype='bar', colors='steelblue',
	edgecolor='green'
)
plt.xlim(xmin=5, xmax=5)

fig.savefig('name')

// Compare to normal
from scipy import stats

density = stats.kde.gaussian_kde(values)
plt.hist(...)
plt.plot(values2, density(values2), ...)

plt.matshow(corr, cmap='RdBu', fignum=fig.number)
plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical')
plt.yticks(range(len(corr.columns)), corr.columns)


import csv

with open('...', 'r') as fp:
	reader = csv.reader(fp, delimiter='...')
	for index, (colum1, colum2, ...) in enumerate(reader):
		
	pd.(read/to)_csv
				_json
				_html
				_clipboard
				_excel
				_hdf
				_feather
				_parquet
				_msgpack
				_stata
				_sas
				_pickie
				_sql
				_gbq

read_csv(
	sep=''
	header= 
	na_value=['', '?', '-']
	use_cols=[]
	names= []
	dtypes= {}
	parse_dates=[0]
	thousands=','
	decimal=','
	skiprows=2
	skip_blank_lines=False
	// to get a Serie (only one column)
	squeeze=True
)

to_csv('...', index=...)

// To get information
pd.read_csv?


import sqlite3

conn = sqlite3.connect('....db')

cur = conn.cursor()
cur.execute('Select ...')
results = cur.fetchall()
df = pd.DataFrame(results)

df = pd.read_sql('Select...', conn)
pd.read_sql_query('...', conn)

// only with sqlAlchemt conn
pd.read_sql_table('...', conn)
 
df.to_sql('db', conn)

cur.close()
conn.close()

https://github.com/krishnatray/RDP-Reading-Data-with-Python-and-Pandas/tree/master/unit-1-reading-data-with-python-and-pandas
