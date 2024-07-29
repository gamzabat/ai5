import tensorflow as tf

import numpy as np
import pandas as pd

from sklearn.datasets import load_wine

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
dataset = load_wine()

x = dataset.data
y = pd.get_dummies(dataset.target)

print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 5555
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

# model.add(Dense(3, activation = 'softmax'))

input1 = Input(shape = (13,))

dense1 = Dense(64, activation = 'relu')(input1)
dense2 = Dense(64, activation = 'relu')(dense1)
dense3 = Dense(64, activation = 'relu')(dense2)
drop1 = Dropout(0.3)(dense3)
dense4 = Dense(64, activation = 'relu')(drop1)
dense5 = Dense(64, activation = 'relu')(dense4)
dense6 = Dense(64, activation = 'relu')(dense5)
drop2 = Dropout(0.3)(dense6)
dense7 = Dense(64, activation = 'relu')(drop2)
dense8 = Dense(64, activation = 'relu')(dense7)
dense9 = Dense(64, activation = 'relu')(dense8)
drop3 = Dropout(0.3)(dense9)
dense10 = Dense(64, activation = 'relu')(drop3)
dense11 = Dense(64, activation = 'relu')(dense10)

output1 = Dense(3, activation = 'softmax')(dense11)

model = Model(inputs = input1, outputs = output1)

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras32/09-wine/'

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

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 64,
    restore_best_weights = True
)

model.fit(
    x_train,
    y_train,
    validation_split = 0.3,
    callbacks = [es, mcp],
    epochs = 1000,
    batch_size = 4,
    verbose = 2
)

end_time = time.time()

loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(np.round(y_pred[:10]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))
print("fit time", "gpu on" if (len(gpus) > 0) else "gpu off", round(end_time - start_time, 2), "초")

# fit time gpu off 2.87 초
# fit time gpu on 29.68 초

# before dropout
# loss : [0.14875008165836334, 0.9444444179534912]
# acc : 0.9444444444444444

# after dropout
# loss : [0.02778332121670246, 1.0]
# acc : 1.0