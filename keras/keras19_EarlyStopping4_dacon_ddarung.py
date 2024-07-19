# 12 dacon copy

# https://dacon.io/competitions/open/235576/overview/description

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

import time

PATH = "./_data/ddarung/" # 상대경로

#1 data
train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(PATH + "submission.csv", index_col = 0)

print(submission_csv) # [715 rows x 1 columns] - 결측치

print(train_csv.shape) # (1459, 10)
print(test_csv.shape) # (715, 9)
print(submission_csv.shape) # (715, 1)

print(train_csv.columns)

# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

#################### 결측치 처리 1. 삭제 ####################

#print(train_csv.isnull().sum())

print(train_csv.isna().sum())

# train_csv = train_csv.dropna()
train_csv = train_csv.dropna()

print(train_csv) # [1328 rows x 10 columns]

print(train_csv.isna().sum())

train_csv.info()
#------------------------------------------------------------
test_csv.info() # 결측치에 평균치를 넣는다 다른 방법은 0으로 채움, 삭제해버리기

test_csv = test_csv.fillna(test_csv.interpolate())

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

print(x)
print(x.shape) # (1328, 9)

y = train_csv['count']

print(y)
print(y.shape) # (1328,)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.80,
    random_state = 4343
)

#2 model
model = Sequential()

model.add(Dense(100, input_dim = 13, activation = 'relu'))

model.add(Dense(100, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))

model.add(Dense(1, activation = 'linear'))

#3 compile
model.compile(loss = 'mse', optimizer = 'adam')

start_time = time.time()

es = EarlyStopping(
    monitor = "val_loss",
    mode = "min",
    patience = 10,
    restore_best_weights = True
)

hist = model.fit(x_train,
                 y_train,
                 validation_split = 0.2,
                 callbacks = [es],
                 epochs = 200,
                 batch_size = 32)

end_time = time.time()

# model은 compile, fit을 통해 최적의 가중치를 가지게 된다

#4 predict
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print(loss)
print(r2) # 2022.5994873046875 : 0.80, 1111, 1000, 32
print("fit time", round(end_time - start_time, 2), "초")

print("========== hist ==========")
print(hist)
print("========== hist.history ==")
print(hist.history)
print("========== loss ==========")
print(hist.history["loss"])
print("========== val_loss ======")
print(hist.history["val_loss"])

plt.rc("font", family = "Gulim")
plt.rc("axes", unicode_minus = False)
plt.figure(figsize = (9, 6))
plt.plot(hist.history["loss"], color = "red", label = "loss")
plt.plot(hist.history["val_loss"], color = "blue", label = "val_loss")
plt.legend(loc = "upper right")
plt.title("따릉이 loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.grid()
plt.show()

y_submit = model.predict(test_csv)

print(y_submit)
print(y_submit.shape) # (715, 1)
#######################################################
# count 컬럼에 값만 넣어주면 된다

submission_csv['count'] = y_submit

print(submission_csv)
print(submission_csv.shape) #(715, 1)

print(loss)

submission_csv.to_csv(PATH + "submission_0719.csv")