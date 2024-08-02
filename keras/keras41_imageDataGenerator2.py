# 배치를 160으로 잡고
# x, y를 추출해서 모델을 만들어서
# acc 0.99 이상

# batch_size = 160

# x_train = xy_train[0][0]
# y_train = xy_train[0][1]

# x_test = xy_test[0][0]
# y_test = xy_test[0][1]

import pandas as pd
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time

#1 data
train_datagen = ImageDataGenerator(
    rescale = 1. / 255,         # 데이터 스케일링
    # horizontal_flip = True,     # 수평 뒤집기
    # vertical_flip = True,       # 수직 뒤집기
    # width_shift_range = 0.1,    # 10% 수평 이동
    # height_shift_range = 0.1,   # 10% 수직 이동
    # rotation_range = 5,         # 정해진 각도로 회전
    # zoom_range = 1.2,           # 축소 또는 확대
    # shear_range = 0.7,          # 수평으로 찌그러트리기
    # fill_mode = 'nearest'       # 이미지가 이동한 후 남은 공간을 어떻게 채우는가
)

test_datagen = ImageDataGenerator(
    rescale = 1. / 255,         # 데이터 스케일링
)

PATH_TRAIN = "./_data/image/brain/train/"
PATH_TEST = "./_data/image/brain/test/"

# 10, 200, 200, 1
xy_train = train_datagen.flow_from_directory(
    PATH_TRAIN,
    target_size = (200, 200),
    batch_size = 160,
    class_mode = 'binary',
    color_mode = 'grayscale',
    shuffle = True
) # Found 160 images belonging to 2 classes.

# 10, 200, 200, 1
xy_test = test_datagen.flow_from_directory(
    PATH_TEST,
    target_size = (200, 200),
    batch_size = 120,
    class_mode = 'binary',
    color_mode = 'grayscale'
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]

x_test = xy_test[0][0]
y_test = xy_test[0][1]

#2 model
model = Sequential()

model.add(Conv2D(32, 3, input_shape = (200, 200, 1), activation = 'relu', padding = 'same'))
model.add(Conv2D(32, 3, activation = 'relu', padding = 'same'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(Conv2D(32, 2, activation = 'relu', padding = 'same'))
model.add(MaxPooling2D())
model.add(Conv2D(16, 2, activation = 'relu', padding = 'same'))
model.add(Flatten())
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

#3 compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras41/imageDataGenerator2/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k41_', date, "_", filename])
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
    patience = 50,
    restore_best_weights = True
)

start_time = time.time()

model.fit(
    x_train,
    y_train,
    validation_split = 0.25,
    callbacks = [es, mcp],
    epochs = 2560,
    batch_size = 160,
    verbose = 1
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))
print("fit time", round(end_time - start_time, 2), "초")