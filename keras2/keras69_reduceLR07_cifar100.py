#35-4 copy

import tensorflow as tf

import pandas as pd
import numpy as np

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

import time

import random as rn

from tensorflow.keras.optimizers import Adam
import tensorflow as tf

rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)

#1 data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# print(np.max(x_train), np.min(x_train))

x_train = x_train.reshape(-1, 32 * 32 * 3)

x_test = x_test.reshape(-1, 32 * 32 * 3)

print(pd.DataFrame(x_train))

x_train = x_train / 255.
x_test = x_test / 255.

print(pd.DataFrame(x_train))

###### 원핫 1-1 케라스
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

###### 원핫 1-2 판다스
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

###### 원핫 1-3 싸이킷런
# y_train = OneHotEncoder(sparse = False).fit_transform(y_train.reshape(-1, 1))
# y_test = OneHotEncoder(sparse = False).fit_transform(y_test.reshape(-1, 1))

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

for learning_rate in lr:
    #2 model
    model = Sequential()

    model.add(Dense(128, input_shape = (32 * 32 * 3, ), activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.25))

    model.add(Dense(100, activation = 'softmax'))

    #3 compile
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = learning_rate), metrics = ['accuracy'])

    ##################### mcp 세이브 파일명 만들기 시작 ###################
    import datetime

    date = datetime.datetime.now()

    print(date) # 2024-07-26 16:49:36.336699
    print(type(date)) # <class 'datetime.datetime'>

    date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

    PATH = './_save/ml05/03-cifar10/'

    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

    filepath = ''.join([PATH, 'ml05_', date, "_", filename])
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
        validation_split = 0.25,
        callbacks = [es, mcp, rlr],
        epochs = 1000,
        batch_size = 256,
        verbose = 1
    )

    #4 predict
    loss = model.evaluate(x_test, y_test, verbose = 0)

    y_pred = model.predict(x_test)

    print("rl: {0}, loss :{1}".format(learning_rate, loss))
    print("acc :", accuracy_score(y_test.argmax(axis = 1), y_pred.argmax(axis = 1)))
    print("fit time", round(time.time() - start_time, 2), "sec")

# lr: 0.1, loss :[4.615432262420654, 0.009999999776482582]
# acc : 0.01
# rl: 0.1, loss :[4.605515003204346, 0.009999999776482582]
# acc : 0.01

# lr: 0.01, loss :[4.60582160949707, 0.009999999776482582]
# acc : 0.01
# rl: 0.01, loss :[4.6058220863342285, 0.009999999776482582]
# acc : 0.01

# lr: 0.005, loss :[4.391546726226807, 0.024299999698996544]
# acc : 0.0243
# rl: 0.005, loss :[4.387573719024658, 0.02239999920129776]
# acc : 0.0224

# lr: 0.001, loss :[4.315570831298828, 0.028999999165534973]
# acc : 0.029
# rl: 0.001, loss :[4.24275016784668, 0.042100001126527786]
# acc : 0.0421

# lr: 0.0005, loss :[3.9033429622650146, 0.09939999878406525]
# acc : 0.0994
# rl: 0.0005, loss :[3.987189531326294, 0.08420000225305557]
# acc : 0.0842

# lr: 0.0001, loss :[3.8207547664642334, 0.1136000007390976]
# acc : 0.1136
# rl: 0.0001, loss :[3.8377890586853027, 0.11029999703168869]
# acc : 0.1103