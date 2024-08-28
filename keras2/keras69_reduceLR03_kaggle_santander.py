# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

import time

import tensorflow as tf
import random as rn

from tensorflow.keras.optimizers import Adam

rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)

#1 data
PATH = "./_data/kaggle/santander/" # 절대경로

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)
test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)
sample_submission_csv = pd.read_csv(PATH + "sample_submission.csv", index_col = 0)

x = train_csv.drop('target', axis = 1)

y = train_csv['target']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.95,
    random_state = 7777,
    stratify = y
)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)

x_train = min_max_scaler.transform(x_train)
x_test = min_max_scaler.transform(x_test)

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

for learning_rate in lr:
    #3 model
    model = Sequential()

    model.add(Dense(256, input_dim = 200, activation = 'relu'))

    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation = 'relu'))

    model.add(Dense(1, activation = 'sigmoid'))

    #3 compile
    model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = learning_rate), metrics = ['accuracy'])

    ##################### mcp 세이브 파일명 만들기 시작 ###################
    import datetime

    date = datetime.datetime.now()

    print(date) # 2024-07-26 16:49:36.336699
    print(type(date)) # <class 'datetime.datetime'>

    date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

    PATH = './_save/keras67/12-kaggle-santander/'

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

    start_time = time.time()

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
        validation_split = 0.1,
        callbacks = [es, mcp, rlr],
        epochs = 1000,
        batch_size = 1024,
        verbose = 1
    )

    end_time = time.time()

    print("fit time :", round(end_time - start_time, 2), "초")

    #4 predict
    loss = model.evaluate(x_test, y_test, verbose = 0)

    y_pred = model.predict(x_test)

    print(np.round(y_pred[:10]))

    print("rl: {0}, loss :{1}".format(learning_rate, loss))
    print("acc :", accuracy_score(y_test, np.round(y_pred)))

    # y_submit = model.predict(test_csv)

    # sample_submission_csv['target'] = np.round(y_submit).astype("int")

    # sample_submission_csv.to_csv(PATH + "sample_submission_0729.csv")

# lr: 0.1, loss :[0.3261803090572357, 0.8995000123977661]
# acc : 0.8995
# rl: 0.1, loss :[0.3261825442314148, 0.8995000123977661]
# acc : 0.8995

# lr: 0.01, loss :[0.22406816482543945, 0.9164000153541565]
# acc : 0.9164
# rl: 0.01, loss :[0.22443969547748566, 0.9164000153541565]
# acc : 0.9164

# lr: 0.005, loss :[0.22361832857131958, 0.9140999913215637]
# acc : 0.9141
# rl: 0.005, loss :[0.22438275814056396, 0.9157000184059143]
# acc : 0.9157

# lr: 0.001, loss :[0.225198894739151, 0.9160000085830688]
# acc : 0.916
# rl: 0.001, loss :[0.2240789532661438, 0.9161999821662903]
# acc : 0.9162

# lr: 0.0005, loss :[0.22323527932167053, 0.9164000153541565]
# acc : 0.9164
# rl: 0.0005, loss :[0.22325053811073303, 0.9162999987602234]
# acc : 0.9163

# lr: 0.0001, loss :[0.22139614820480347, 0.9178000092506409]
# acc : 0.9178
# rl: 0.0001, loss :[0.22136318683624268, 0.9168000221252441]
# acc : 0.9168