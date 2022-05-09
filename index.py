#%%
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

#%%
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v2.feature_column as feature_column
from IPython.display import clear_output
from six.moves import urllib

#%%
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') #/data/titanic/train.csv
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') #/data/titanic/eval.csv
y_train = dftrain.pop('deck')
y_eval = dfeval.pop('deck')

#%%
# .head() show the first five items in dataframe
print(dftrain.head())
print()

# .describe() statical analysis of our data
print(dftrain.describe())
print()

# .shape --> (row, coloumn)
print(dftrain.shape)
print()

#%%
dftrain.age.hist(bins=20)

#%%
dftrain['class'].value_counts().plot(kind='barh')

# %%
