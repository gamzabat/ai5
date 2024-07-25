# 18 bike1 copy

# https://www.kaggle.com/competitions/bike-sharing-demand

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from keras.callbacks import EarlyStopping

import time

#1 data
PATH = "C:/ai5/_data/kaggle/bike-sharing-demand/" # 절대경로

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)
test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)
sampleSubmission_csv = pd.read_csv(PATH + "sampleSubmission.csv", index_col = 0)

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

print(train_csv.shape, test_csv.shape, sampleSubmission_csv.shape) # (10886, 11) (6493, 8) (6493, 1)
print(train_csv.columns) # ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']

train_csv.info()
test_csv.info()

print(train_csv.describe().T) # 50% : 전체의 가운데 값

########################## 결측치 확인 ##########################
print(train_csv.isna().sum())

print(test_csv.isna().sum())
#---------------------------------------------------------------
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1) # [] -> python에서 list 형식이다

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.800,
    random_state = 7777 
)

# min_max_scaler = MinMaxScaler()

# min_max_scaler.fit(x_train)

# x_train = min_max_scaler.transform(x_train)
# x_test = min_max_scaler.transform(x_test)
# test_csv = min_max_scaler.transform(test_csv)

# stardard_scaler = StandardScaler().fit(x_train)

# x_train = stardard_scaler.transform(x_train)
# x_test = stardard_scaler.transform(x_test)
# test_csv = stardard_scaler.transform(test_csv)

# max_abs_scaler = MaxAbsScaler().fit(x_train)

# x_train = max_abs_scaler.transform(x_train)
# x_test = max_abs_scaler.transform(x_test)
# test_csv = max_abs_scaler.transform(test_csv)

robust_scaler = RobustScaler().fit(x_train)

x_train = robust_scaler.transform(x_train)
x_test = robust_scaler.transform(x_test)
test_csv = robust_scaler.transform(test_csv)

#2 model
model = Sequential()

model.add(Dense(100, input_dim = 13, activation = 'relu'))

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

start_time = time.time()

es = EarlyStopping(
    monitor = "val_loss",
    mode = 'min', # 모르면 auto
    patience = 50,
    restore_best_weights = True
)

hist = model.fit(x_train,
          y_train,
          validation_split = 0.2,
          callbacks = [es],
          epochs = 500,
          batch_size = 32,
          verbose = 1)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("rs :", r2)
print("fit time :", round(end_time - start_time, 2), "초")

# 20513.6875 : mse 0.80 7777 300 128

y_submit = model.predict(test_csv)

print(y_submit)
print(y_submit.shape)

sampleSubmission_csv['count'] = np.round(y_submit).astype("int")

print(sampleSubmission_csv)
print(sampleSubmission_csv.shape)

sampleSubmission_csv.to_csv(PATH + "sampleSubmission_0725.csv")

# before scaler
# loss : 5440.0703125
# rs : 0.8349684860953358

# after minMaxScaler
# loss : 1448.45458984375
# rs : 0.9560592736058223

# after standardScaler
# loss : 1555.6107177734375
# rs : 0.9528085599443087

# after minAbsScaler
# loss : 3188.50146484375
# rs : 0.9032727255687395

# after robustScaler
# loss : 1634.345703125
# rs : 0.9504200299235311