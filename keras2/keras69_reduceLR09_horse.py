import pandas as pd
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.decomposition import PCA

import time

import random as rn

from tensorflow.keras.optimizers import Adam
import tensorflow as tf

rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)

WIDTH = 25
HEIGHT = 25

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

test_datagen = ImageDataGenerator(
    rescale = 1. / 255,         # 데이터 스케일링
)

PATH_TRAIN = "./_data/image/horse_human/"

start_time = time.time()

xy_train = train_datagen.flow_from_directory(
    PATH_TRAIN,
    target_size = (WIDTH, HEIGHT),
    batch_size = 1000,
    class_mode = 'binary',
    color_mode = 'rgb',
    shuffle = True
)

lead_time_train = time.time() - start_time

print(lead_time_train)

start_time = time.time()

x_train, x_test, y_train, y_test = train_test_split(
    xy_train[0][0],
    xy_train[0][1],
    train_size = 0.8,
    shuffle = False
)

lead_time_split = time.time() - start_time

print(lead_time_split)

x_train = x_train.reshape(-1, WIDTH * HEIGHT * 3)
x_test = x_test.reshape(-1, WIDTH * HEIGHT * 3)

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

for learning_rate in lr:
    #2 model
    model = Sequential()

    model.add(Dense(128, input_shape = (WIDTH * HEIGHT * 3, ), activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.25))

    model.add(Dense(1, activation = 'sigmoid'))

    #3 compile
    model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = learning_rate), metrics = ['accuracy'])

    ##################### mcp 세이브 파일명 만들기 시작 ###################
    import datetime

    date = datetime.datetime.now()

    print(date) # 2024-07-26 16:49:36.336699
    print(type(date)) # <class 'datetime.datetime'>

    date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

    PATH = './_save/keras67/imageDataGenerator4/'

    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

    filepath = ''.join([PATH, 'k67_', date, "_", filename])
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

    start_time = time.time()

    model.fit(
        x_train,
        y_train,
        validation_split = 0.25,
        callbacks = [es, mcp, rlr],
        epochs = 2560,
        batch_size = 32,
        verbose = 1
    )

    end_time = time.time()

    #4 predict
    loss = model.evaluate(x_test, y_test, verbose = 0)

    y_pred = model.predict(x_test)

    print("lead split time :", lead_time_split)
    print("rl: {0}, loss :{1}".format(learning_rate, loss))
    print("acc :", accuracy_score(y_test, np.round(y_pred)))
    print("fit time", round(end_time - start_time, 2), "초")

# lead split time : 4.832361936569214
# loss : [0.011994075961411, 0.9950000047683716]
# acc : 0.995
# fit time 64.14 초

# lr: 0.1, loss :[0.6932957768440247, 0.5]
# acc : 0.5
# rl: 0.1, loss :[0.6880201697349548, 0.5600000023841858]
# acc : 0.56

# lr: 0.01, loss :[0.693260908126831, 0.5]
# acc : 0.5
# rl: 0.01, loss :[0.2515236735343933, 0.9150000214576721]
# acc : 0.915

# lr: 0.005, loss :[0.6932452321052551, 0.5]
# acc : 0.5
# rl: 0.005, loss :[0.2711719274520874, 0.9300000071525574]
# acc : 0.93

# lr: 0.001, loss :[0.6848092675209045, 0.5608000159263611]
# acc : 0.5608
# rl: 0.001, loss :[0.2105873078107834, 0.925000011920929]
# acc : 0.925

# lr: 0.0005, loss :[0.6735682487487793, 0.5659999847412109]
# acc : 0.566
# rl: 0.0005, loss :[0.240895614027977, 0.9350000023841858]
# acc : 0.935

# lr: 0.0001, loss :[0.671410322189331, 0.5771999955177307]
# acc : 0.5772
# rl: 0.0001, loss :[0.1807451993227005, 0.9449999928474426]
# acc : 0.945