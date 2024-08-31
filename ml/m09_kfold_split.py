import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
import pandas as pd

# data
datasets = load_iris()

df = pd.DataFrame(datasets.data, columns = datasets.feature_names)

n_splits = 3

kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 3333)

for train_index, val_index in kfold.split(df):
    print('=================================================')
    print(train_index, '\n', val_index)
    print('number of train data :', len(train_index))
    print('number of validation data :', len(val_index))