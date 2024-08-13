import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time

#1 data
dataset = load_diabetes()

x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 7777
)

x_train = x_train.reshape(-1, 5, 2, 1)
x_test = x_test.reshape(-1, 5, 2, 1)

#2 model
model = Sequential()

# model.add(Conv2D(64, 2, input_shape = (5, 2, 1), activation = 'relu', padding = 'same'))
# model.add(Conv2D(64, 2, activation = 'relu', padding = 'same'))
# model.add(Conv2D(64, 2, activation = 'relu', padding = 'same'))

model.add(LSTM(units = 64, input_shape = (5, 2), return_sequences = True, activation = 'relu'))
model.add(LSTM(units = 64, return_sequences = True, activation = 'relu'))
model.add(LSTM(units = 64, activation = 'relu'))

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

PATH = './_save/keras59/03-diabetes/'

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
    batch_size = 256,
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

# loss : [2771.278076171875, 0.0] - CNN
# rs : 0.569203862250742
# fit time 25.48 sec

# loss : [4056.263671875, 0.0] - LSTM
# rs : 0.3694524911488243
# fit time 40.47 sec