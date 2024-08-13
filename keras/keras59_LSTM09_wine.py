import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import r2_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import time

#1 data
dataset = load_wine()

x = dataset.data
y = pd.get_dummies(dataset.target)

print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 5555
)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)

x_train = min_max_scaler.transform(x_train)
x_test = min_max_scaler.transform(x_test)

x_train = x_train.reshape(-1, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)

#2 model
model = Sequential()

# model.add(Conv2D(64, 2, input_shape = (13, 1, 1), activation = 'relu', padding = 'same'))
# model.add(Conv2D(64, 2, activation = 'relu', padding = 'same'))
# model.add(Conv2D(64, 2, activation = 'relu', padding = 'same'))

model.add(LSTM(units = 64, input_shape = (13, 1), return_sequences = True, activation = 'relu'))
model.add(LSTM(units = 64, return_sequences = True, activation = 'relu'))
model.add(LSTM(units = 64, activation = 'relu'))

model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.summary()

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras59/09-wine/'

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
    patience = 16,
    restore_best_weights = True
)

start_time = time.time()

hist = model.fit(
    x_train,
    y_train,
    validation_split = 0.2,
    callbacks = [es, mcp],
    epochs = 2560,
    batch_size = 512,
    verbose = 1
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(y_test)
print(y_pred)

print("loss :", loss)
print("acc :", accuracy_score(y_test.values.argmax(axis = 1), y_pred.argmax(axis = 1)))
print("fit time", round(end_time - start_time, 2), "sec")

# loss : [0.11855915188789368, 0.9722222089767456] - CNN
# acc : 0.9722222222222222
# fit time 4.02 sec

# loss : [0.2604152262210846, 0.9166666865348816] - LSTM
# acc : 0.9166666666666666
# fit time 14.46 sec