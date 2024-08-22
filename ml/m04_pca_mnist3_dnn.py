#35-4 copy

import tensorflow as tf

import pandas as pd
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

import time

from sklearn.decomposition import PCA

#1 data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,) 

# print(np.max(x_train), np.min(x_train))

x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

print(pd.DataFrame(x_train))

x_train = x_train / 255.
x_test = x_test / 255.

print(pd.DataFrame(x_train))

###### 원핫 1-1 케라스
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

###### 원핫 1-2 판다스
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

###### 원핫 1-3 싸이킷런
# y_train = OneHotEncoder(sparse = False).fit_transform(y_train.reshape(-1, 1))
# y_test = OneHotEncoder(sparse = False).fit_transform(y_test.reshape(-1, 1))

n_components = np.array([154, 331, 486, 713, 764])

for i in reversed(n_components):
    pca = PCA(n_components = i)

    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    #2 model
    model = Sequential()

    model.add(Dense(128, input_shape = (x_train.shape[1], ), activation = 'relu'))
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

    model.add(Dense(10, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    ##################### mcp 세이브 파일명 만들기 시작 ###################
    import datetime

    date = datetime.datetime.now()

    print(date) # 2024-07-26 16:49:36.336699
    print(type(date)) # <class 'datetime.datetime'>

    date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

    PATH = './_save/ml04/'

    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

    filepath = ''.join([PATH, 'ml04_', date, "_", filename])
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
        validation_split = 0.25,
        callbacks = [es, mcp],
        epochs = 1000,
        batch_size = 256,
        verbose = 1
    )

    #4 predict
    loss = model.evaluate(x_test, y_test, verbose = 0)

    y_pred = model.predict(x_test)

    print("loss :", loss)
    print("acc :", accuracy_score(y_test.values.argmax(axis = 1), y_pred.argmax(axis = 1)))
    print("fit time", round(time.time() - start_time, 2), "sec")

# loss : [0.12391505390405655, 0.9757000207901001]
# acc : 0.9757
# fit time 36.69 sec

# n_components = 784
# loss : [0.12570573389530182, 0.9725000262260437]
# acc : 0.9725
# fit time 13.96 sec

# n_components = 713
# loss : [0.12951946258544922, 0.9714000225067139]
# acc : 0.9714
# fit time 14.02 sec

# n_components = 486
# loss : [0.11334796249866486, 0.9742000102996826]
# acc : 0.9742
# fit time 16.46 sec

# n_components = 331
# loss : [0.1248810663819313, 0.9717000126838684]
# acc : 0.9717
# fit time 15.66 sec

# n_components = 154
# loss : [0.09649935364723206, 0.9779999852180481]
# acc : 0.978
# fit time 25.06 sec