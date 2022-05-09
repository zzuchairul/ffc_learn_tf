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
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#%%
CATEGORICAL_COLOUMS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']

NUMBERICAL_COLOUMS = ['age', 'fare']

feature_coloums = []
# %%
for feature_name in CATEGORICAL_COLOUMS:
    vocl = dftrain[feature_name].unique()
    feature_coloums.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocl))

for feature_name in NUMBERICAL_COLOUMS:
    feature_coloums.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_coloums)
# %%
