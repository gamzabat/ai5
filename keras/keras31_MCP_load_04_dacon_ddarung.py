# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd

from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

PATH = "./_data/dacon/ddarung/" # 상대경로

#1 data
train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(PATH + "submission.csv", index_col = 0)

print(submission_csv) # [715 rows x 1 columns] - 결측치

train_csv = train_csv.dropna()

train_csv.info()

test_csv.info() # 결측치에 평균치를 넣는다 다른 방법은 0으로 채움, 삭제해버리기

test_csv = test_csv.fillna(test_csv.mean())

test_csv.info()

train_dt = pd.DatetimeIndex(train_csv.index)

train_csv['day'] = train_dt.day
train_csv['month'] = train_dt.month
train_csv['year'] = train_dt.year
train_csv['hour'] = train_dt.hour
train_csv['dow'] = train_dt.dayofweek

test_dt = pd.DatetimeIndex(test_csv.index)

test_csv['day'] = test_dt.day
test_csv['month'] = test_dt.month
test_csv['year'] = test_dt.year
test_csv['hour'] = test_dt.hour
test_csv['dow'] = test_dt.dayofweek

train_csv.info()

x = train_csv.drop(['count'], axis = 1)

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.80,
    random_state = 4343
)

#2 model
print("======================= MCP 출력 ==========================")
model = load_model('./_save/keras30_mcp/04-dacon-ddarung/k30_240726_190354_0149-3168.4036.hdf5')

#4 predict
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("rs :", r2)

y_submit = model.predict(test_csv)

#######################################################
# count 컬럼에 값만 넣어주면 된다
submission_csv['count'] = np.round(y_submit).astype("int")

submission_csv.to_csv(PATH + "submission_0726.csv")