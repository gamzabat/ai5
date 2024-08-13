import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time

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

x_train = x_train.reshape(-1, 31, 3, 1)
x_test = x_test.reshape(-1, 31, 3, 1)

#2 model
model = Sequential()

# model.add(Conv2D(128, 3, input_shape = (31, 3, 1), activation = 'relu', padding = 'same'))
# model.add(BatchNormalization())
# model.add(Conv2D(128, 3, activation = 'relu', padding = 'same'))
# model.add(Dropout(0.25))
# model.add(MaxPooling2D())
# model.add(BatchNormalization())
# model.add(Conv2D(128, 3, activation = 'relu', padding = 'same'))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
# model.add(Conv2D(64, 2, activation = 'relu', padding = 'same'))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(64, activation = 'relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.25))
# model.add(Dense(9, activation = 'softmax'))

model.add(LSTM(units = 128, input_shape = (31, 3), return_sequences = True, activation = 'relu'))
model.add(BatchNormalization())
model.add(LSTM(units = 128, return_sequences = True, activation = 'relu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(LSTM(units = 128, return_sequences = True, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(LSTM(units = 64, return_sequences = True, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(LSTM(units = 64, activation = 'relu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(9, activation = 'softmax'))

model.summary()

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras59/13-kaggle-otto/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k59_', date, "_", filename])
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

start_time = time.time()

hist = model.fit(
    x_train,
    y_train,
    validation_split = 0.2,
    callbacks = [es, mcp],
    epochs = 2560,
    batch_size = 2048,
    verbose = 1
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print("loss :", loss)
print("acc :", accuracy_score(y_test.idxmax(axis = 1), y_pred.argmax(axis = 1)))
print("fit time", round(end_time - start_time, 2), "sec")

# loss : [0.5116978287696838, 0.8021169900894165] - CNN
# acc : 0.8021170006464124
# fit time 38.12 sec

# loss : [2.025235176086426, 0.3101971447467804] - LSTM
# acc : 0.3101971557853911
# fit time 128.55 sec