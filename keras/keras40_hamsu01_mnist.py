import tensorflow as tf

import pandas as pd
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

import time

#1 data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,) 

###### scaling 1-1
# x_train = x_train / 255.
# x_test = x_test / 255.

###### scaling 1-2
# x_train = (x_train - 127.5) / 127.5
# x_test = (x_test - 127.5) / 127.5

###### scaling 2. MinMaxScaling()
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

scaler = MinMaxScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

print(np.max(x_train), np.min(x_train))

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

###### 원핫 1-1 케라스
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

###### 원핫 1-2 판다스
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

###### 원핫 1-3 싸이킷런
# y_train = OneHotEncoder(sparse = False).fit_transform(y_train.reshape(-1, 1))
# y_test = OneHotEncoder(sparse = False).fit_transform(y_test.reshape(-1, 1))

#2 model
# model = Sequential()

# model.add(Conv2D(64, (3, 3), input_shape = (28, 28, 1), activation = 'relu', padding = 'same'))
# model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.3))
# model.add(Conv2D(32, (2, 2), activation = 'relu', padding = 'same'))
# model.add(MaxPooling2D())

# model.add(Conv2D(32, (2, 2), activation = 'relu', padding = 'same'))
# model.add(Flatten())
# model.add(Dense(32, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))

# model.add(Dense(10, activation = 'softmax'))

# model.summary()

input1 = Input(shape = (28, 28, 1))

conv2D1 = Conv2D(64, 3, activation = 'relu', padding = 'same')(input1)
conv2D2 = Conv2D(64, 3, activation = 'relu', padding = 'same')(conv2D1)
maxPooling2D1 = MaxPooling2D()(conv2D2)
dropout1 = Dropout(0.30)(maxPooling2D1)
conv2D3 = Conv2D(32, 2, activation = 'relu', padding = 'same')(dropout1)
maxPooling2D2 = MaxPooling2D()(conv2D3)
conv2D4 = Conv2D(32, 2, activation = 'relu', padding = 'same')(maxPooling2D2)
flatten1 = Flatten()(conv2D4)
dense1 = Dense(32, activation = 'relu')(flatten1)
dense2 = Dense(16, activation = 'relu')(dense1)

output1 = Dense(10, activation = 'softmax')(dense2)

model = Model(inputs = input1, outputs = output1)

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras35/mnist/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k35_', date, "_", filename])
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
    patience = 16,
    restore_best_weights = True
)

start_time = time.time()

model.fit(
    x_train,
    y_train,
    validation_split = 0.2,
    callbacks = [es, mcp],
    epochs = 2560,
    batch_size = 128,
    verbose = 1
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(y_test.values)
print(y_pred)

print(y_test.values.shape)
print(y_pred.shape)

print("loss :", loss)
print("acc :", accuracy_score(y_test.values.argmax(axis = 1), y_pred.argmax(axis = 1)))
print("fit time", round(end_time - start_time, 2), "초")

# loss : [0.030024154111742973, 0.9919999837875366]
# acc : 0.992
# fit time 76.4 초