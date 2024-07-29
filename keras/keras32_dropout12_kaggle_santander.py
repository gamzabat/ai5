# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time

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
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras32/12-kaggle-santander/'

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
    patience = 16,
    restore_best_weights = True
)

model.fit(
    x_train,
    y_train,
    validation_split = 0.1,
    callbacks = [es, mcp],
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

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))

y_submit = model.predict(test_csv)

print(y_submit[:10])

sample_submission_csv['target'] = np.round(y_submit).astype("int")

sample_submission_csv.to_csv(PATH + "sample_submission_0729.csv")

# before dropout
# loss : [0.22236484289169312, 0.9161999821662903]
# acc : 0.9162

# after dropout
# loss : [0.22476914525032043, 0.9142000079154968]
# acc : 0.9142