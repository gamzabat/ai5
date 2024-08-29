# https://dacon.io/competitions/open/235576/overview/description

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from keras.callbacks import EarlyStopping, ModelCheckpoint

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
######################## Population 로그변환 ############################
# y_train = np.log1p(y_train)
# y_test = np.log1p(y_test)
######################## Population 로그변환 ############################

#2 model
model = Sequential()

model.add(Dense(64, input_dim = 13, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation = 'relu'))

model.add(Dense(1))

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
    patience = 32,
    restore_best_weights = True
)

hist = model.fit(x_train,
                 y_train,
                 validation_split = 0.2,
                 callbacks = [es, mcp],
                 epochs = 1024,
                 batch_size = 8)

#4 predict
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("rs :", r2)

# before log transform
# loss : 3147.33447265625
# rs : 0.4763130545906934

# after log transform
# loss : 0.6863811612129211
# rs : 0.26694299319585657