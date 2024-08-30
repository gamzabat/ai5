# https://www.kaggle.com/competitions/otto-group-product-classification-challenge

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, cross_val_score

import xgboost as xgb

#1 data
PATH = "./_data/kaggle/otto/" # 절대경로

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

x = train_csv.drop('target', axis = 1)

le = LabelEncoder()

train_csv['target'] = le.fit_transform(train_csv['target'])

print(train_csv['target'])

y = train_csv['target']

kfold = KFold(n_splits = 5, shuffle = True, random_state = 333)

# model
model = xgb.XGBRegressor()

# train
scores = cross_val_score(
    model,
    x,
    y,
    cv = kfold,
    verbose = 1
)

print('accuracy :', scores, '\n average :', np.round(np.mean(scores), 4))

# accuracy : [0.72223164 0.73857016 0.74008921 0.73086639 0.73008739] 
# average : 0.7324