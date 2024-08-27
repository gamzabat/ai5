# https://www.kaggle.com/datasets/stytch16/jena-climate-2009-2016/data

# y = T degC

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, BatchNormalization, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import Sequence

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.metrics import RootMeanSquaredError

from sklearn.decomposition import PCA

import time

import os

import random as rn

from tensorflow.keras.optimizers import Adam
import tensorflow as tf

rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(gpus[0], True)

#1 data
PATH_LOAD_CSV = "./_data/kaggle/jena/"
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
dataset = pd.read_csv(PATH_LOAD_CSV + "jena_climate_2009_2016_with_datetime.csv", index_col = 0)

# train_dt = pd.DatetimeIndex(dataset.index)

# dataset['day'] = train_dt.day
# dataset['month'] = train_dt.month
# dataset['year'] = train_dt.year
# dataset['hour'] = train_dt.hour
# dataset['dow'] = train_dt.dayofweek

# #--------------------------------------------
scaler = MaxAbsScaler()

temper = dataset['T (degC)']

dataset = dataset.drop(["wv (m/s)","max. wv (m/s)","wd (deg)"], axis = 1)

dataset = scaler.fit_transform(dataset)
#--------------------------------------------
x_data = dataset[:-288]

y_data = temper[144:-144].values

print(x_data.shape, y_data.shape)

x_predict = dataset[-288:-144].reshape(1, 144 * 16)

answer = temper[-144:].values

x = split_dataset(x_data, 144)
y = split_dataset(y_data, 144)

# print('-------------- data ready -------------------')

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.9,
    random_state = 7777
)

x_train = x_train.reshape(-1, 144 * 16)
x_test = x_test.reshape(-1, 144 * 16)

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

for learning_rate in lr:
    model = Sequential()

    model.add(Dense(128, input_shape = (144 * 16, )))
    model.add(Dropout(0.25))
    model.add(Dense(128))
    model.add(Dropout(0.25))
    model.add(Dense(128))
    model.add(Dropout(0.25))
    model.add(Dense(128))
    model.add(Dropout(0.25))
    model.add(Dense(128))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Dropout(0.25))
    model.add(Dense(32))
    model.add(Dense(144))

    #3 compile, fit
    model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

    #################### mcp 세이브 파일명 만들기 시작 ###################
    import datetime

    date = datetime.datetime.now()

    date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

    PATH = './_save/ml05/kaggle_jena/'

    filename = '{epoch:04d}_{val_loss:.8f}.hdf5'

    filepath = ''.join([PATH, 'ml05_', date, "_", filename])
    #################### mcp 세이브 파일명 만들기 끝 ###################

    mcp = ModelCheckpoint(
        monitor = 'val_loss',
        mode = 'auto',
        verbose = 1,
        save_best_only = True,
        filepath = filepath
    )

    es = EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        patience = 16,
        restore_best_weights = True
    )

    start_time = time.time()

    model.fit(
        x_train,
        y_train,
        callbacks = [es, mcp],
        epochs = 10000,
        batch_size = 512,
        validation_split = 0.1,
#        validation_data = (x_predict, answer.reshape(1, 144)),
        verbose = 1
    )

    end_time = time.time()

    results = model.evaluate(x_test, y_test, batch_size = 3000)

    # model = load_model('./_save/keras55/jena.hdf5')

    y_predict = model.predict(x_predict, batch_size = 3000)

    print("lr: {0}, loss :{1}".format(learning_rate, results))

    print("pred :", y_predict)

    rmse = RMSE(answer, y_predict.reshape(144, 1))

    print("rmse :", rmse)

# rmse : 1.5734169717470299

# lr: 0.1, loss :[7.230741500854492, 0.041654765605926514]

# lr: 0.01, loss :[7.222434997558594, 0.04286870360374451]

# lr: 0.005, loss :[7.225479602813721, 0.04160716012120247]

# lr: 0.001, loss :[7.312013626098633, 0.043749403208494186]

# lr: 0.0005, loss :[7.507391452789307, 0.041868988424539566]

# lr: 0.0001, loss :[7.402224540710449, 0.04101209342479706]