import tensorflow as tf

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_covtype

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import time

gpus = tf.config.experimental.list_physical_devices('GPU')

print(gpus)

#1 data
dataset = fetch_covtype()

x = pd.DataFrame(dataset.data).iloc[:, :13]

y = pd.get_dummies(dataset.target)

print(y.head())

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 7777,
    stratify = y
)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)

x_train = min_max_scaler.transform(x_train)
x_test = min_max_scaler.transform(x_test)

#2 model
# model = Sequential()

# model.add(Dense(64, input_dim = 13, activation = 'relu'))

# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))

# model.add(Dense(7, activation = 'softmax'))

input1 = Input(shape = (13,))

dense1 = Dense(64, activation = 'relu')(input1)
dense2 = Dense(64, activation = 'relu')(dense1)
dense3 = Dense(64, activation = 'relu')(dense2)
dense4 = Dense(64, activation = 'relu')(dense3)
drop1 = Dropout(0.3)(dense4)
dense5 = Dense(64, activation = 'relu')(drop1)
dense6 = Dense(64, activation = 'relu')(dense5)
dense7 = Dense(64, activation = 'relu')(dense6)
drop2 = Dropout(0.3)(dense7)
dense8 = Dense(64, activation = 'relu')(drop2)
dense9 = Dense(64, activation = 'relu')(dense8)
dense10 = Dense(64, activation = 'relu')(dense9)
drop3 = Dropout(0.3)(dense5)
dense11 = Dense(64, activation = 'relu')(drop3)
dense12 = Dense(64, activation = 'relu')(dense11)

output1 = Dense(7, activation = 'softmax')(dense12)

model = Model(inputs = input1, outputs = output1)

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras32/10-fetch-covtype/'

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
    monitor = 'val_loss',
    mode = 'min',
    patience = 8,
    restore_best_weights = True
)

start_time = time.time()

model.fit(
    x_train,
    y_train,
    validation_split = 0.25,
    callbacks = [es, mcp],
    epochs = 512,
    batch_size = 4096,
    verbose = 1
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(np.round(y_pred[:10]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))
print("fit time", "gpu on" if (len(gpus) > 0) else "gpu off", round(end_time - start_time, 2), "초")

# fit time gpu on 100.53 초
# fit time gpu off 67.44 초

# before dropout
# loss : [0.3247015178203583, 0.8664836287498474]
# acc : 0.8614579658012271

# after dropout
# loss : [0.6058399081230164, 0.7236559987068176]
# acc : 0.7068922489092364