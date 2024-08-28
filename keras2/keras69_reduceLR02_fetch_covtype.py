import numpy as np
import pandas as pd

from sklearn.datasets import fetch_covtype

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import numpy as np

from sklearn.decomposition import PCA

import tensorflow as tf
import random as rn

from tensorflow.keras.optimizers import Adam

rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)

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

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

for learning_rate in lr:
    #2 model
    model = Sequential()

    model.add(Dense(64, input_dim = 13, activation = 'relu'))

    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))

    model.add(Dense(7, activation = 'softmax'))

    #3 compile
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = learning_rate), metrics = ['accuracy'])

    ##################### mcp 세이브 파일명 만들기 시작 ###################
    import datetime

    date = datetime.datetime.now()

    print(date) # 2024-07-26 16:49:36.336699
    print(type(date)) # <class 'datetime.datetime'>

    date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

    PATH = './_save/keras67/10-fetch-covtype/'

    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

    filepath = ''.join([PATH, 'keras67_', date, "_", filename])
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
        patience = 10,
        restore_best_weights = True
    )

    rlr = ReduceLROnPlateau(
        monitor = 'val_loss',
        mode = 'auto',
        patience = 5,
        verbose = 1,
        factor = 0.8
    )

    model.fit(
        x_train,
        y_train,
        validation_split = 0.25,
        callbacks = [es, mcp, rlr],
        epochs = 512,
        batch_size = 1024,
        verbose = 1
    )

    #4 predict
    loss = model.evaluate(x_test, y_test, verbose = 0)

    y_pred = model.predict(x_test)

    print("rl: {0}, loss :{1}".format(learning_rate, loss))
    print("acc :", accuracy_score(y_test, np.round(y_pred)))

# before dropout
# loss : [0.3247015178203583, 0.8664836287498474]
# acc : 0.8614579658012271

# after dropout
# loss : [0.6058399081230164, 0.7236559987068176]
# acc : 0.7068922489092364

# lr: 0.1, loss :[1.2052181959152222, 0.4876036047935486]
# acc : 0.0
# rl: 0.1, loss :[1.2053791284561157, 0.4876036047935486]
# acc : 0.0

# lr: 0.01, loss :[0.7268705368041992, 0.6936481595039368]
# acc : 0.6733991377158938
# rl: 0.01, loss :[0.6953397989273071, 0.7183205485343933]
# acc : 0.7095513885183687

# lr: 0.005, loss :[0.6212747693061829, 0.7379930019378662]
# acc : 0.7248608039379362
# rl: 0.005, loss :[0.5731644034385681, 0.762665331363678]
# acc : 0.7477775961033708

# lr: 0.001, loss :[0.4898867607116699, 0.7893255949020386]
# acc : 0.7754102734008589
# rl: 0.001, loss :[0.5400922298431396, 0.7678803205490112]
# acc : 0.7532593822878927

# lr: 0.0005, loss :[0.35641714930534363, 0.8565183281898499]
# acc : 0.846845606395704
# rl: 0.0005, loss :[0.45289433002471924, 0.7999104857444763]
# acc : 0.7881982392881423

# lr: 0.0001, loss :[0.4295538067817688, 0.8229907751083374]
# acc : 0.8100909615070179
# rl: 0.0001, loss :[0.4069572389125824, 0.83658766746521]
# acc : 0.8284295586172474