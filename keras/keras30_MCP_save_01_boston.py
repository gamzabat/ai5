import sklearn as sk

print(sk.__version__) # 0.24.2

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.datasets import load_boston

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import time

# data
dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.7,
    random_state = 7777
)

# model
model = Sequential()

model.add(Dense(10, input_dim = 13, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))

model.add(Dense(1))

# compile
model.compile(loss = 'mse', optimizer = 'adam')

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras30_mcp/01-boston/'

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

start_time = time.time()

hist = model.fit(x_train,
          y_train,
          validation_split = 0.2,
          epochs = 1000,
          callbacks = [mcp],
          batch_size = 32,
          verbose = 1)

end_time = time.time()

# predict
loss = model.evaluate(x_test, y_test, verbose = 0)

print(loss)

y_predict = model.predict(x_test)

print(y_predict)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("r2 :", r2)
print("fit time :", round(end_time - start_time, 2), "초")