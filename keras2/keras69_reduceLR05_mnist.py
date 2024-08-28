import tensorflow as tf

import pandas as pd
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

import time

import random as rn

from tensorflow.keras.optimizers import Adam

rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)

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

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

for learning_rate in lr:
    #2 model
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape = (28, 28, 1), activation = 'relu', padding = 'same'))
    model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.3))
    model.add(Conv2D(32, (2, 2), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (2, 2), activation = 'relu', padding = 'same'))
    model.add(Flatten())
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))

    model.add(Dense(10, activation = 'softmax'))

    model.summary()

    #3 compile
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = learning_rate), metrics = ['accuracy'])

    ##################### mcp 세이브 파일명 만들기 시작 ###################
    import datetime

    date = datetime.datetime.now()

    print(date) # 2024-07-26 16:49:36.336699
    print(type(date)) # <class 'datetime.datetime'>

    date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

    PATH = './_save/keras67/mnist/'

    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

    filepath = ''.join([PATH, 'k67_', date, "_", filename])
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

    rlr = ReduceLROnPlateau(
        monitor = 'val_loss',
        mode = 'auto',
        patience = 5,
        verbose = 1,
        factor = 0.8
    )

    start_time = time.time()

    model.fit(
        x_train,
        y_train,
        validation_split = 0.2,
        callbacks = [es, mcp, rlr],
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

    print("rl: {0}, loss :{1}".format(learning_rate, loss))
    print("acc :", accuracy_score(y_test.values.argmax(axis = 1), y_pred.argmax(axis = 1)))
    print("fit time", round(end_time - start_time, 2), "초")

# lr: 0.1, loss :[2.3019144535064697, 0.11349999904632568]
# acc : 0.1135
# rl: 0.1, loss :[2.3016245365142822, 0.11349999904632568]
# acc : 0.1135

# lr: 0.01, loss :[0.042992979288101196, 0.9861000180244446]
# acc : 0.9861

# lr: 0.005, loss :[0.040421221405267715, 0.989300012588501]
# acc : 0.9893
# rl: 0.005, loss :[0.027915973216295242, 0.9922999739646912]
# acc : 0.9923

# lr: 0.001, loss :[0.03013009950518608, 0.9905999898910522]
# acc : 0.9906
# rl: 0.001, loss :[0.030093597248196602, 0.9908999800682068]
# acc : 0.9909

# lr: 0.0005, loss :[0.02917785570025444, 0.991100013256073]
# acc : 0.9911
# rl: 0.005, loss :[0.027915973216295242, 0.9922999739646912]
# acc : 0.9923

# lr: 0.0001, loss :[0.029909441247582436, 0.9898999929428101]
# acc : 0.9899
# rl: 0.0001, loss :[0.031615015119314194, 0.9902999997138977]
# acc : 0.9903