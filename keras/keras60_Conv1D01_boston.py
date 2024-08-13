import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LSTM, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time

#1 data
dataset = load_boston()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 7777
)

x_train = x_train.reshape(-1, 13, 1, 1)
x_test = x_test.reshape(-1, 13, 1, 1)

#2 model
model = Sequential()

# model.add(Conv2D(64, 2, input_shape = (13, 1, 1), activation = 'relu', padding = 'same'))
# model.add(Conv2D(64, 2, activation = 'relu', padding = 'same'))
# model.add(Conv2D(64, 2, activation = 'relu', padding = 'same'))

model.add(Conv1D(filters = 10, kernel_size = 3, input_shape = (13, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1))

model.summary()

#3 compile
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras60/01-boston/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k60_', date, "_", filename])
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
print("rs :", r2_score(y_test, y_pred))
print("fit time", round(end_time - start_time, 2), "sec")

# loss : [18.914005279541016, 0.0] - CNN
# rs : 0.7675890325318571
# fit time 18.74 sec

# loss : [31.667612075805664, 0.0] - LSTM
# rs : 0.6108756010878105
# fit time 9.89 sec

# loss : [71.11067962646484, 0.0] - Conv1D
# rs : 0.12620828835415498
# fit time 4.27 sec