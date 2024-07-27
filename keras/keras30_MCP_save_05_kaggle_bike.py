# https://www.kaggle.com/competitions/bike-sharing-demand

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from keras.callbacks import EarlyStopping, ModelCheckpoint

#1 data
PATH = "./_data/kaggle/bike-sharing-demand/" # 절대경로

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

train_csv.info()
test_csv.info()
#---------------------------------------------------------------
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1) # [] -> python에서 list 형식이다

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.800,
    random_state = 7777 
)

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

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras30_mcp/05-kaggle-bike/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k30_', date, "_", filename])
##################### mcp 세이브 파일명 만들기 끝 ###################

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1,
    save_best_only = True,
    filepath = filepath
)

es = EarlyStopping(
    monitor = "val_loss",
    mode = 'min', # 모르면 auto
    patience = 50,
    restore_best_weights = True
)

hist = model.fit(x_train,
          y_train,
          validation_split = 0.2,
          callbacks = [es, mcp],
          epochs = 500,
          batch_size = 32,
          verbose = 1)

#4 predict
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("rs :", r2)

y_submit = model.predict(test_csv)

sampleSubmission_csv['count'] = np.round(y_submit).astype("int")

sampleSubmission_csv.to_csv(PATH + "sampleSubmission_0726.csv")