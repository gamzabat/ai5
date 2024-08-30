# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score

import xgboost as xgb

#1 data
PATH = "./_data/kaggle/santander/" # 절대경로

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

x = train_csv.drop('target', axis = 1)

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

# accuracy : [0.20892856 0.19532253 0.2017563  0.1999986  0.20333786] 
# average : 0.2019