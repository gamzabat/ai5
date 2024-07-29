# https://www.kaggle.com/competitions/otto-group-product-classification-challenge

import tensorflow as tf

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time

gpus = tf.config.experimental.list_physical_devices('GPU')

print(gpus)

#1 data
PATH = "./_data/kaggle/otto/" # 절대경로

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)
test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)
sample_submission_csv = pd.read_csv(PATH + "sampleSubmission.csv", index_col = 0)

x = train_csv.drop('target', axis = 1)
#x = train_csv

le = LabelEncoder()

train_csv['target'] = le.fit_transform(train_csv['target'])

print(train_csv['target'])

y = pd.get_dummies(train_csv['target'])
#y = train_csv['target']

print(y)

print(train_csv.isna().sum())
print(test_csv.isna().sum())

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

#3 model
model = Sequential()

model.add(Dense(128, input_dim = 93, activation = 'relu'))

model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))

model.add(Dense(9, activation = 'softmax'))

# input1 = Input(shape = (93,))

# dense1 = Dense(128, activation = 'relu')(input1)
# dense2 = Dense(128, activation = 'relu')(dense1)
# dense3 = Dense(128, activation = 'relu')(dense2)
# drop1 = Dropout(0.2)(dense3)
# dense4 = Dense(64, activation = 'relu')(drop1)
# dense5 = Dense(64, activation = 'relu')(dense4)
# drop2 = Dropout(0.2)(dense5)
# dense6 = Dense(32, activation = 'relu')(drop2)
# dense7 = Dense(32, activation = 'relu')(dense6)

# output1 = Dense(9, activation = 'softmax')(dense7)

# model = Model(inputs = input1, outputs = output1)

input_layer = Input(batch_shape = model.layers[0].input_shape)

prev_layer = input_layer

for layer in model.layers:
    prev_layer = layer(prev_layer)

model = Model([input_layer], [prev_layer])

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras32/13-kaggle-otto/'

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
    patience = 32,
    restore_best_weights = True
)

start_time = time.time()

model.fit(
    x_train,
    y_train,
    validation_split = 0.25,
    callbacks = [es, mcp],
    epochs = 5120,
    batch_size = 2048,
    verbose = 2
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

# print(np.round(y_pred[:10]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))
print("fit time", "gpu on" if (len(gpus) > 0) else "gpu off", round(end_time - start_time, 2), "초")

y_submit = model.predict(test_csv)

print(y_submit[:10])

for i in range(9):
    sample_submission_csv['Class_' + str(i + 1)] = y_submit[:, i]

sample_submission_csv.to_csv(PATH + "sampleSubmission_0729.csv")

# fit time gpu on 7.59 초
# fit time gpu off 9.55 초

# before dropout
# loss : [0.5492791533470154, 0.7920976281166077]
# acc : 0.743939883645766

# after dropout
# loss : [0.5500587821006775, 0.7950872778892517]
# acc : 0.7581609566903684