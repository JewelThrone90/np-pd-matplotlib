##### numpy #####
import numpy as np
# numpy defaults to rows: 1d is a row vector, matrices filled row first:
np.arange(10) # shape = (,1)
np.arange(4).reshape(2,2) # fills row first

# broadcasting matches whichever dimensions possible
m = np.arange(12).reshape(3,4) # 3x4 matrix
m + np.arange(4) # recycle row first
m + np.arange(3).reshape(-1,1) # recycle col first

# array methods include: m.sum, m.mean, m.max, m.argmax()
m.argmax(axis = 0) # specify the dimension to dissapear, 0 = row

# argmax gets dataposition by default, use ravel_index to get coordinate
m.argmax() # element number
np.unravel_index(m.argmax(), m.shape) #coords

# use np.where to convert boolean index into integer positions
m[np.where(m >5)] # same as m[m>5]

# transpose and matrix multiplication
m.T # transpose matrix
m.T.dot(m) # matrix multiplication

# speed: reshaping and slicing is fast, 'fancy indexing' is slow
m.reshape(4,3) #fast, provides a 'view of the 'data', data is not copied.
m[1:,2:] #fast, provides a 'view of the 'data', data is not copied.
m[m>2] # slow, data is copied in memory

# flatten: copies data, ravel: does not copy data, if elements are continuous
# e.g. you haven't transposed or sliced
b = m.flatten() # safe, changin b will not effect m
b = m.ravel() # efficient, but changing b effect m

# linspace vs arange
np.linspace(0,10,21) # spcificy number of return values
np.arange(0,10,0.5) # specify step size, excludes endpoint

##### matplotlib #####







##### pandas #####
import pandas as pd
#dealing with large data sets: slice, shape, col summaries, plots

# start with sample data:
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# first things
df.shape #shape of dataframe
df.dtypes # column data types
df.info() # info on each column, object can be string or anything e.g a list
df.head() # gives top rows
df.columns # get column names
df.index # get row names e.g. dates, integers
df.values # gets values as a numpy array e.g. to put in scikit learn

# selection from data frame (columns)
df['sepal_length'] # get column as pandas series (extension of numpy array)
df[['sepal_length']] # forces pandas to return a data frame
df.drop(['sepal_length', 'sepal_width'], axis = 1) #drop columns (axis = 1)
df.sepal_length # works if referring to one column
# inplace vs by value, many methods allow you to change a df in place 
# e.g. df.drop(['sepal_width'], inplace = True)

# selecting rows: 
# loc does a string match against index names, iloc does index values
df.loc[0] # get first row as a pd series
df.loc[[0,1,2]] # get first three rows as data frame
df.loc[-1] # doesn't work because no row is called '-1'
df.iloc[[1,2,-1]] # this works as normal

# selecting columns and rows
df.loc[:, ['sepal_length', 'sepal_width']] # all rows for chosen columns
df.iloc[1:10,[2,4]] # subset using index location
# boolean selection
df.loc[df.species == 'setosa', :] # select just sertosa, all columns
# combining booleans requires (brackets) around each term 
df.loc[(df.species == 'setosa') & (df.petal_width > 0.1)] # works
df.loc[df.species == 'setosa' & df.petal_width > 0.1] # error

#add columns by referring to name as if it existed
df['mean_value'] = df.iloc[:,0:4].mean(axis = 1) #mean value column

# grouping data i.e. 1) split, 2) apply, 3) combine
df.groupby(['species'])['sepal_width'].mean() # mean for each species
# more generic aggregation, function input in .agg() can be any aggregation
# function i.e. something that takes multiple values and returns 1
df.groupby(['species'])['sepal_width'].agg(np.std) # numpy std fn
# multiple columns can be provided
df.groupby(['species', 'sepal_length'])[['sepal_width']].agg(len)
# add on reset_index() to turn a grouped df back into a standard df
df.groupby(['species', 'sepal_length'])[['sepal_width']].agg(len).reset_index()

# data processing, melted data can easily be reformatted or plotted
df.melt(id_vars='species') # convert to molten state
billboard = pd.read_csv('https://raw.githubusercontent.com/chendaniely/scipy-2019-pandas/master/data/billboard.csv')
# piping using dots, whole statement must be enclose in (brackets)
(
 billboard
    .melt(id_vars = ['year', 'artist', 'track', 'time', 'date.entered'],
          value_name ='rank',
          var_name='week')
    .groupby('artist')['rank']
    .mean()
)
# df.pivot_table can be used to reverse pivoting

# apply: functions that take one value apply element wise
# aggregate functions apply column wise 
df.iloc[:,0:4].apply(lambda x: x**2) #apply fn to each element
df.species.apply(lambda x: x.upper()) # capatilize species name
df.iloc[:,0:4].apply(lambda x: np.mean(x)**2) # apply aggregate fn to each col

# plotting

# quick plotting methods
df.plot() # plot line graph for numeric variables
df.plot(kind = 'hist') # histogram
# bar chart for categorical data: convert using value_counts() first
df.species.value_counts().plot(kind = 'bar') # bar plot for categorical data

# seaborn is closer to ggplot in R
import seaborn as sns
sns.distplot(df.sepal_width) # empircal pdf + histogram
#linear regression plot split by species
sns.lmplot(x = 'sepal_width', y = 'sepal_length', 
           hue = 'species', data = df)
# use row = cat1 and col = cat2 to get seperate plots by category 1 and 2
# facet grids can be calculated directly use sns.FacetGrid

# using subplots with matplot lib
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2) # fig is whole thing, axes is a subplot
ax1.scatter(df.sepal_width, df.sepal_length)
ax2.hist(df.sepal_width)
fig.show()

# seaborn figures can also be plot on axes, check the fn has an ax argument
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2) # fig is whole thing, axes is a subplot
sns.distplot(df.sepal_width, ax = ax1)
# lmplot doesn't have an ax argument, but regplot does
sns.regplot(df.sepal_width, df.sepal_length, ax = ax2) 
fig.tight_layout()
fig.show()

# convert categorical variables into dummies (needed for sklearn)
pd.get_dummies(df) # converts all categorical data in data frame
pd.get_dummies(df, drop_first = True) # technically first dummy can be ignored




