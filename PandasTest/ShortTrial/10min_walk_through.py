import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# time series
s = pd.Series([1,3,5,np.nan,6,8])
s

# DataFrame with datetime and labeled columns
dates = pd.date_range('20130101', periods=6)    # increase by one day
dates
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list(['A', 'B', 'C', 'D']))    # or list('ABCD')
df

# Create DF by passing a dict of objects that be converted to series-like
df2 = pd.DataFrame({    'A' : 1.,
                        'B' : pd.Timestamp('20130102'),
                        'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                        'D' : np.array([3] * 4, dtype='int32'),
                        'E' : pd.Categorical(['test', 'train', 'test', 'train']),
                        'F' : 'foo'})
df2
df2.get_values()[2,4]
# Out[12]: 'test'
df2.get_value(2, 'D')
# Out[17]: 3
df2.dtypes
# Out[19]:
# A           float64
# B    datetime64[ns]
# C           float32
# D             int32
# E          category
# F            object
# dtype: object

df.head()
df.tail(3)
# Furthermore, index, columns, values, describe(), T
df.sort_values(by='B')
df.sort_values(by='B', ascending=False)

# Plotting
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()
# Plot all of the columns with labels
df = pd.DataFrame(np.random.randn(1000, 4), index=ts.index, columns=['A', 'B', 'C', 'D'])
df = df.cumsum()
plt.figure(); df.plot(); plt.legend(loc='best')

# Write to cvs
df.to_csv('./foo.csv')

# HDF5
df.to_hdf('foo.h5','df')    # ImportError: HDFStore requires PyTables, "No module named tables" problem importing
pd.read_hdf('foo.h5','df')