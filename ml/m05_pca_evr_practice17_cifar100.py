#35-4 copy

import tensorflow as tf

import pandas as pd
import numpy as np

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

import time

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

n_components = np.array([202, 659, 1481, 3072])

for i in reversed(n_components):
    pca = PCA(n_components = i)

    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    #2 model
    model = Sequential()

    model.add(Dense(128, input_shape = (i, ), activation = 'relu'))
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
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

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

    print("n_components =", i)
    print("loss :", loss)
    print("acc :", accuracy_score(y_test.argmax(axis = 1), y_pred.argmax(axis = 1)))
    print("fit time", round(time.time() - start_time, 2), "sec")

# loss : [4.35453987121582, 0.03060000017285347]
# acc : 0.0306
# fit time 33.1 sec

# n_components = 3072
# loss : [3.8632383346557617, 0.09700000286102295]
# acc : 0.097

# n_components = 1481
# loss : [3.84574294090271, 0.10140000283718109]
# acc : 0.1014

# n_components = 659
# loss : [3.7252707481384277, 0.12110000103712082]
# acc : 0.1211

# n_components = 202
# loss : [3.51959228515625, 0.16899999976158142]
# acc : 0.169