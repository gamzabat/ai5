# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score

import xgboost as xgb

PATH = "./_data/kaggle/playground-series-s4e1/" # 상대경로

le = LabelEncoder()

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

le.fit(train_csv['Geography'])

train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])

le.fit(train_csv['Gender'])

train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

submission_csv = pd.read_csv(PATH + "sample_submission.csv", index_col = 0)

print(submission_csv) # [110023 rows x 1 columns]

train_csv.info() # CustomerId Surname  CreditScore Geography Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited

train_csv = train_csv.drop(['CustomerId', 'Surname'], axis = 1)

train_csv.info()

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis = 1)

test_csv.info()

x = train_csv.drop(['Exited'], axis = 1)

y = train_csv['Exited']

print(y.value_counts())

kfold = KFold(n_splits = 5, shuffle = True, random_state = 333)

# model
model = xgb.XGBClassifier()

# train
scores = cross_val_score(
    model,
    x,
    y,
    cv = kfold,
    verbose = 2
)

print('accuracy :', scores, '\n average :', np.round(np.mean(scores), 4))

# accuracy : [0.86469537 0.86296846 0.86651316 0.86190808 0.86420651]
# average : 0.8641