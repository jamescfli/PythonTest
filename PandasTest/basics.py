import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2 types: series and dataframe
foo = pd.read_csv('PandasTest/foo.csv', header=0)
print foo.head(5)
print foo.tail(5)
print foo.columns
print len(foo)      # 10000

pd.options.display.float_format = '{:,.3f}'.format
print foo.describe()    # class: pandas.core.frame.DataFrame
print foo['A'].shape    # (1000,)
# where foo['A'] === foo.A
print foo.A[3]          # indexing

# plot data frame
foo.plot(x=np.arange(1000), y=['A', 'B', 'C', 'D'])

# save
foo.to_csv('PandasTest/foo_saved.csv')  # saved with index