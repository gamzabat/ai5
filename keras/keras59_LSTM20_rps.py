import pandas as pd
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input, BatchNormalization, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time

#1 data
train_datagen = ImageDataGenerator(
    rescale = 1. / 255,           # 데이터 스케일링
    # horizontal_flip = True,     # 수평 뒤집기
    # vertical_flip = True,       # 수직 뒤집기
    # width_shift_range = 0.1,    # 10% 수평 이동
    # height_shift_range = 0.1,   # 10% 수직 이동
    # rotation_range = 5,         # 정해진 각도로 회전
    # zoom_range = 1.2,           # 축소 또는 확대
    # shear_range = 0.7,          # 수평으로 찌그러트리기
    # fill_mode = 'nearest'       # 이미지가 이동한 후 남은 공간을 어떻게 채우는가
)

PATH_TRAIN = "./_data/image/rps/"

start_time = time.time()

xy_train = train_datagen.flow_from_directory(
    PATH_TRAIN,
    target_size = (100, 100),
    batch_size = 10000,
    class_mode = 'categorical',    # 다중분류 - 원핫되어서 결과가 나온다
#    class_mode = 'sparse',         # 다중분류
#    class_mode = 'binary',         # 이진분류
#    class_mode = None,             # y값이 없다
    color_mode = 'rgb'
)

lead_time_train = time.time() - start_time

print(lead_time_train)

start_time = time.time()

y = xy_train[0][1]

x_train, x_test, y_train, y_test = train_test_split(
    xy_train[0][0],
    y,
    train_size = 0.8,
    stratify = y,
    random_state = 7777
)

lead_time_split = time.time() - start_time

print(lead_time_split)

x_train = x_train.reshape(-1, 300, 100)
x_test = x_test.reshape(-1, 300, 100)

#2 model
model = Sequential()

model.add(LSTM(units = 32, input_shape = (300, 100), return_sequences = True, activation = 'relu'))
model.add(Flatten()) 
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu')) 
model.add(Dropout(0.5))
model.add(Dense(3, activation = 'softmax'))

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras59/imageDataGenerator5/'

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
    patience = 5,
    restore_best_weights = True
)

start_time = time.time()

model.fit(
    x_train,
    y_train,
    validation_split = 0.1,
    callbacks = [es, mcp],
    epochs = 2560,
    batch_size = 512,
    verbose = 1
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(y_test[:20])
print(y_pred[:20])

print("lead split time :", lead_time_split)
print("loss :", loss)
print("acc :", accuracy_score(y_test.argmax(axis = 1), y_pred.argmax(axis = 1)))
print("fit time", round(end_time - start_time, 2), "초")

# loss : [0.003939483314752579, 1.0] - CNN
# acc : 1.0
# fit time 105.35 초

# loss : [1.0986155271530151, 0.3333333432674408] - LSTM
# acc : 0.3333333333333333
# fit time 17.8 초