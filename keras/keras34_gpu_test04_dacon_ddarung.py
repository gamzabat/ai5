# https://dacon.io/competitions/open/235576/overview/description

import tensorflow as tf

import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from keras.callbacks import EarlyStopping, ModelCheckpoint

import time

gpus = tf.config.experimental.list_physical_devices('GPU')

print(gpus)

PATH = "./_data/dacon/ddarung/" # 상대경로

#1 data
train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(PATH + "submission.csv", index_col = 0)

print(submission_csv) # [715 rows x 1 columns] - 결측치

train_csv = train_csv.dropna()

print(train_csv) # [1328 rows x 10 columns]

print(train_csv.isna().sum())

train_csv.info()
#------------------------------------------------------------
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

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)

x_train = min_max_scaler.transform(x_train)
x_test = min_max_scaler.transform(x_test)
test_csv = min_max_scaler.transform(test_csv)

#2 model
# model = Sequential()

# model.add(Dense(64, input_dim = 13, activation = 'relu'))

# model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation = 'relu'))

# model.add(Dense(1))

input1 = Input(shape = (13,))

dense1 = Dense(64, activation = 'relu')(input1)
dense2 = Dense(64, activation = 'relu')(dense1)
drop1 = Dropout(0.3)(dense2)
dense3 = Dense(64, activation = 'relu')(drop1)
dense4 = Dense(64, activation = 'relu')(dense3)
drop2 = Dropout(0.3)(dense4)
dense5 = Dense(64, activation = 'relu')(drop2)
dense6 = Dense(32, activation = 'relu')(dense5)
dense7 = Dense(32, activation = 'relu')(dense6)
drop3 = Dropout(0.3)(dense7)
dense8 = Dense(32, activation = 'relu')(drop3)
dense9 = Dense(32, activation = 'relu')(dense8)
drop4 = Dropout(0.3)(dense9)
dense10 = Dense(32, activation = 'relu')(drop4)
output1 = Dense(1)(dense10)

model = Model(inputs = input1, outputs = output1)

#3 compile
model.compile(loss = 'mse', optimizer = 'adam')

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras32/04-dacon-ddarung/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k32_', date, "_", filename])
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
    mode = "min",
    patience = 128,
    restore_best_weights = True
)

start_time = time.time()

hist = model.fit(x_train,
                 y_train,
                 validation_split = 0.2,
                 callbacks = [es, mcp],
                 epochs = 1024,
                 batch_size = 2048)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("rs :", r2)
print("fit time", "gpu on" if (len(gpus) > 0) else "gpu off", round(end_time - start_time, 2), "초")

y_submit = model.predict(test_csv)
#######################################################
# count 컬럼에 값만 넣어주면 된다

submission_csv['count'] = y_submit

print(submission_csv)
print(submission_csv.shape) #(715, 1)

# print(loss)

submission_csv.to_csv(PATH + "submission_0729.csv")

# fit time gpu on 7.27 초
# fit time gpu off 4.5 초