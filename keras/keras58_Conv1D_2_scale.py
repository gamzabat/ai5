import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU, BatchNormalization, Bidirectional, Conv1D, Flatten

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

x = np.array([[1, 2, 3],
              [2, 3, 4],
              [3, 4, 5],
              [4, 5, 6],
              [5, 6, 7],
              [6, 7, 8],
              [7, 8, 9],
              [8, 9, 10],
              [9, 10, 11],
              [10, 11, 12],
              [20, 30, 40],
              [30, 40, 50],
              [40, 50, 60]])

y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

#2 model
model = Sequential()

model.add(Conv1D(filters = 10, kernel_size = 3, input_shape = (3, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='relu'))

#3 compile, fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras58/conv1d2/'

filename = '{epoch:04d}_{val_loss:.8f}.hdf5'

filepath = ''.join([PATH, 'k58_', date, "_", filename])
##################### mcp 세이브 파일명 만들기 끝 ###################

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
    patience = 32,
    restore_best_weights = True
)

model.fit(
    x,
    y,
    validation_data = (np.array([50, 60, 70]).reshape(1, 3, 1), np.array([[80.0]])),
    callbacks = [mcp],
    epochs = 500,
    batch_size = 3,
    verbose = 1
)

results = model.evaluate(x, y)

print("loss :", results)

# model = load_model('./_save/k52_lstm_scale2.hdf5')

x_predict = np.array([50, 60, 70]).reshape(1, 3, 1)

y_pred = model.predict(x_predict)

print("pred :", y_pred)

# pred : [[80.57921]]

# pred : [[80.006775]] : Bidirectional

# pred : [[79.96994]] : Conv1D

# pred : [[80.75793]] : Conv1D