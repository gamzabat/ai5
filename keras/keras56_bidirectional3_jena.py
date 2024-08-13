# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

# y = T degC

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, BatchNormalization, Dropout, Flatten, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.metrics import RootMeanSquaredError

import time

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(gpus[0], True)

#1 data
PATH_LOAD_CSV = "c:/ai5/_data/kaggle/jena/"
#=================================================================
def split_dataset(dataset, size):
    result = []

    for i in range(len(dataset) - size + 1):
        if (i % 10000 == 0):
            print(i)

        subset = dataset[i : (i + size)]

        result.append(subset)

    return np.array(result)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
#=================================================================
dataset = pd.read_csv(PATH_LOAD_CSV + "jena_climate_2009_2016.csv", index_col = 0)

train_dt = pd.DatetimeIndex(dataset.index)

dataset['day'] = train_dt.day
dataset['month'] = train_dt.month
dataset['year'] = train_dt.year
dataset['hour'] = train_dt.hour
dataset['dow'] = train_dt.dayofweek

# #--------------------------------------------
scaler = MaxAbsScaler()

temper = dataset['T (degC)']

dataset = dataset.drop(["wv (m/s)","max. wv (m/s)","wd (deg)"], axis = 1)

dataset = scaler.fit_transform(dataset)
# #--------------------------------------------
x_data = dataset[:-288]

y_data = temper[144:-144].values

# print(x_data.shape, y_data.shape)

x_predict = dataset[-288:-144].reshape(1, 144, 16)

answer = temper[-144:].values

x = split_dataset(x_data, 144)
y = split_dataset(y_data, 144)

# print('-------------- data ready -------------------')

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.9,
    random_state = 2532
)

model = Sequential()

model.add(Bidirectional(GRU(units=64, activation = 'tanh'), input_shape=(144, 16)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(144))

# #3 compile, fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy', RootMeanSquaredError(name='rmse')])

# #################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras55/kaggle_jena/'

filename = '{epoch:04d}_{val_rmse:.8f}.hdf5'

filepath = ''.join([PATH, 'k55_', date, "_", filename])
# #################### mcp 세이브 파일명 만들기 끝 ###################

mcp = ModelCheckpoint(
    monitor = 'val_rmse',
    mode = 'auto',
    verbose = 1,
    save_best_only = True,
    filepath = filepath
)

es = EarlyStopping(
    monitor = 'val_rmse',
    mode = 'min',
    patience = 32,
    restore_best_weights = True
)

start_time = time.time()

model.fit(
    x_train,
    y_train,
    callbacks = [es, mcp],
    epochs = 10000,
    batch_size = 2048,
    validation_data = (x_predict, answer.reshape(1, 144)),
    verbose = 1
)

end_time = time.time()

results = model.evaluate(x_test, y_test, batch_size = 3000)

y_predict = model.predict(x_predict, batch_size = 3000)

# ===================== 결과 csv 저장 ========================
submit = pd.read_csv(PATH_LOAD_CSV + "jena_climate_2009_2016.csv")

print(submit)

submit = submit[['Date Time', 'T (degC)']].tail(144)

submit['T (degC)'] = y_predict.reshape(144, 1)

PATH_TO_CSV = "c:/ai5/_data/_save/keras55/"

submit.to_csv(PATH_TO_CSV + 'jena.csv', index = False)
# ===========================================================

print("pred :", y_predict)

rmse = RMSE(answer, y_predict.reshape(144, 1))

print("rmse :", rmse)

# rmse : 1.731790642096893