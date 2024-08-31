from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd

#1 data
PATH = "./_data/kaggle/otto/" # 절대경로

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

x = train_csv.drop('target', axis = 1)

le = LabelEncoder()

train_csv['target'] = le.fit_transform(train_csv['target'])

print(train_csv['target'])

y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle = True,
    random_state = 3333,
    train_size = 0.8
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

n_splits = 5

kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 333)

# model
model = xgb.XGBRegressor()

# train
scores = cross_val_score(
    model,
    x_train,
    y_train,
    cv = kfold
) # 기준 점수 확인

print('accuracy :', scores, '\n average :', np.round(np.mean(scores), 4))

y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

print(y_predict)
print(y_test)

acc = r2_score(y_test, y_predict)

print('cross_val_predict ACC :', acc)

# cross_val_predict ACC : 0.6905988279753593