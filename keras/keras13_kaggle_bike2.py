# 기존 캐글 데이터에서

#1 train_csv의 y를 casual과 register로 잡는다
#  그래서 훈련을 하여 test_csv의 casual과 register를 predict한다

#2 test_csv에 casual과 register 컬럼을 합쳐서

#3 train_csv에 y를 count로 잡는다

#4 전체 훈련

#5 test_csv 예측하여 submission에 붙인다

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1 data
PATH = "./_data/bike-sharing-demand/" # 절대경로

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)
test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)
#test_csv = pd.read_csv(PATH + "test2.csv", index_col = 0)
sampleSubmission_csv = pd.read_csv(PATH + "sampleSubmission.csv", index_col = 0)

# train_dt = pd.DatetimeIndex(train_csv.index)

# train_csv['day'] = train_dt.day
# train_csv['month'] = train_dt.month
# train_csv['year'] = train_dt.year
# train_csv['hour'] = train_dt.hour
# train_csv['dow'] = train_dt.dayofweek

# test_dt = pd.DatetimeIndex(test_csv.index)

# test_csv['day'] = test_dt.day
# test_csv['month'] = test_dt.month
# test_csv['year'] = test_dt.year
# test_csv['hour'] = test_dt.hour
# test_csv['dow'] = test_dt.dayofweek

print(train_csv.shape, test_csv.shape, sampleSubmission_csv.shape) # (10886, 11) (6493, 8) (6493, 1)
print(train_csv.columns) # ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']

train_csv.info()
test_csv.info()

print(train_csv.describe().T) # 50% : 전체의 가운데 값

########################## 결측치 확인 ##########################
print(train_csv.isna().sum())

print(test_csv.isna().sum())
# -------------------------------------------------------------------------------------
# x_pre = train_csv.drop(['casual', 'registered', 'count'], axis = 1)

# y_pre = train_csv[['casual', 'registered']]

# x_train_pre, x_test_pre, y_train_pre, y_test_pre = train_test_split(
#     x_pre,
#     y_pre,
#     train_size = 0.800,
#     random_state = 7777 
# )
# # -------------------------------------------------------------------------------------
# model_pre = Sequential()

# model_pre.add(Dense(100, input_dim = 13, activation = 'relu'))

# model_pre.add(Dense(100, activation = 'relu'))
# model_pre.add(Dense(90, activation = 'relu'))
# model_pre.add(Dense(80, activation = 'relu'))
# model_pre.add(Dense(70, activation = 'relu'))
# model_pre.add(Dense(60, activation = 'relu'))
# model_pre.add(Dense(50, activation = 'relu'))
# model_pre.add(Dense(40, activation = 'relu'))
# model_pre.add(Dense(30, activation = 'relu'))
# model_pre.add(Dense(20, activation = 'relu'))
# model_pre.add(Dense(10, activation = 'relu'))

# model_pre.add(Dense(2, activation = 'linear'))
# # -------------------------------------------------------------------------------------
# model_pre.compile(loss = 'mse', optimizer = 'adam')

# model_pre.fit(x_train_pre, y_train_pre, epochs = 1000, batch_size = 64)

# loss_pre = model_pre.evaluate(x_test_pre, y_test_pre)

# print(loss_pre)
# # -------------------------------------------------------------------------------------
# y_submit_pre = model_pre.predict(test_csv)

# print(y_submit_pre)

# test_csv[['casual', 'registered']] = y_submit_pre

# print(type(test_csv))
# print(type(y_submit_pre))

# test_csv.to_csv(PATH + "test2.csv")
# -------------------------------------------------------------------------------------
x = train_csv.drop(['count'], axis = 1) # [] -> python에서 list 형식이다

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.800,
    random_state = 7777
)

#2 model
model = Sequential()

model.add(Dense(100, input_dim = 15, activation = 'relu'))

model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))

model.add(Dense(1, activation = 'linear'))

#3 compile
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train, y_train, epochs = 200, batch_size = 64)

#4 predict
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

y_submit = model.predict(test_csv)

print(y_submit)
print(y_submit.shape)

sampleSubmission_csv['count'] = y_submit

print(sampleSubmission_csv)
print(sampleSubmission_csv.shape)

print(r2)
print(loss)

# 100.46796417236328 : mae 0.80 7777 200 64
# 20513.6875 : mse 0.80 7777 300 128

sampleSubmission_csv.to_csv(PATH + "sampleSubmission_0718.csv")