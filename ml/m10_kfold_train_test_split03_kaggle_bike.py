from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score
import pandas as pd

#1 data
PATH = "./_data/kaggle/bike-sharing-demand/" # 절대경로

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)
test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

sampleSubmission_csv = pd.read_csv(PATH + "sampleSubmission.csv", index_col = 0)
#---------------------------------------------------------------
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1) # [] -> python에서 list 형식이다

y = train_csv['count']

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
model = SVR()

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

# cross_val_predict ACC : 0.12359174970227216