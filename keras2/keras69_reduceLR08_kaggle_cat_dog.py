import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.decomposition import PCA

import time

import os

import random as rn

from tensorflow.keras.optimizers import Adam
import tensorflow as tf

rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

WIDTH = 25
HEIGHT = 25

#1 data
train_datagen = ImageDataGenerator(
    rescale = 1. / 255,         # 데이터 스케일링
    horizontal_flip = True,     # 수평 뒤집기
    vertical_flip = True,       # 수직 뒤집기
    width_shift_range = 0.1,    # 10% 수평 이동
    height_shift_range = 0.1,   # 10% 수직 이동
    rotation_range = 5,         # 정해진 각도로 회전
    zoom_range = 1.2,           # 축소 또는 확대
    shear_range = 0.7,          # 수평으로 찌그러트리기
    fill_mode = 'nearest'       # 이미지가 이동한 후 남은 공간을 어떻게 채우는가
)

test_datagen = ImageDataGenerator(
    rescale = 1. / 255,         # 데이터 스케일링
)

PATH_TRAIN = "./_data/kaggle/cat_dog/train"
PATH_SUBMIT = "./_data/kaggle/cat_dog/test"

PATH_SUBMISSION = "./_data/kaggle/cat_dog/"

sample_submission_csv = pd.read_csv(PATH_SUBMISSION + "sample_submission.csv", index_col = 0)

start_time = time.time()

xy_train = train_datagen.flow_from_directory(
    PATH_TRAIN,
    target_size = (WIDTH, HEIGHT),
    batch_size = 25000,
    class_mode = 'binary',
    color_mode = 'rgb',
    shuffle = True
)

lead_time_train = time.time() - start_time

print(lead_time_train)

start_time = time.time()

xy_test = test_datagen.flow_from_directory(
    directory = PATH_SUBMIT,
    target_size = (WIDTH, HEIGHT),
    batch_size = 12500,
    class_mode = 'binary',
    color_mode = 'rgb',
    shuffle = False
)

lead_time_test = time.time() - start_time

print(lead_time_test)

x = xy_train[0][0]
y = xy_train[0][1]

start_time = time.time()

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.9,
    stratify = y,
    random_state = 7777
)

lead_time_split = time.time() - start_time

print(lead_time_split)

x_train = x_train.reshape(-1, WIDTH * HEIGHT * 3)
x_test = x_test.reshape(-1, WIDTH * HEIGHT * 3)

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

for learning_rate in lr:
    #2 model
    model = Sequential()

    model.add(Dense(128, input_shape = (25 * 25 * 3, ), activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation = 'relu'))

    model.add(Dense(1, activation = 'sigmoid'))

    #3 compile
    model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = learning_rate), metrics = ['accuracy'])

    ##################### mcp 세이브 파일명 만들기 시작 ###################
    import datetime

    date = datetime.datetime.now()

    date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

    PATH = './_save/ml05/kaggle_cat_dog/'

    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

    filepath = ''.join([PATH, 'ml05_', date, "_", filename])
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
        validation_split = 0.1,
        callbacks = [es, mcp, rlr],
        epochs = 512,
        batch_size = 64,
        verbose = 1
    )

    end_time = time.time()

    #4 predict
    loss = model.evaluate(x_test, y_test, verbose = 0, batch_size = 16)

    y_pred = model.predict(x_test, batch_size = 16)

    print("lead split time :", lead_time_split)
    print("rl: {0}, loss :{1}".format(learning_rate, loss))
    print("acc :", accuracy_score(y_test, np.round(y_pred)))
    print("fit time", round(end_time - start_time, 2), "sec")

# y_submit = model.predict(xy_test[0][0], batch_size = 16)

# sample_submission_csv['label'] = y_submit

# sample_submission_csv.to_csv(PATH_SUBMISSION + "sampleSubmission_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".csv")

# lr: 0.1, loss :[0.6933776140213013, 0.5]
# acc : 0.5
# rl: 0.1, loss :[0.6931486129760742, 0.5]
# acc : 0.5

# lr: 0.01, loss :[0.693260908126831, 0.5]
# acc : 0.5
# rl: 0.01, loss :[0.6933249235153198, 0.5]
# acc : 0.5

# lr: 0.005, loss :[0.6933416724205017, 0.5]
# acc : 0.5
# rl: 0.005, loss :[0.6932452917098999, 0.5]
# acc : 0.5

# lr: 0.001, loss :[0.6750475764274597, 0.5748000144958496]
# acc : 0.5748
# rl: 0.001, loss :[0.6874405741691589, 0.5479999780654907]
# acc : 0.548

# lr: 0.0005, loss :[0.6712384819984436, 0.5748000144958496]
# acc : 0.5748
# rl: 0.0005, loss :[0.6738184094429016, 0.5555999875068665]
# acc : 0.5556

# lr: 0.0001, loss :[0.6719509959220886, 0.5839999914169312]
# acc : 0.584
# rl: 0.0001, loss :[0.6702492237091064, 0.5807999968528748]
# acc : 0.5808