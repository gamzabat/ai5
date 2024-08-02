import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

WIDTH = 100
HEIGHT = 100

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

xy_train1 = train_datagen.flow_from_directory(
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
    class_mode = None,
    color_mode = 'rgb',
    shuffle = False
)

lead_time_test = time.time() - start_time

print(lead_time_test)

x = xy_train1[0][0]
y = xy_train1[0][1]

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

#2 model
model = Sequential()

model.add(Conv2D(128, 3, input_shape = (WIDTH, HEIGHT, 3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(128, 3, activation = 'relu', padding = 'same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Conv2D(128, 3, activation = 'relu', padding = 'same'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Conv2D(64, 2, activation = 'relu', padding = 'same'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Conv2D(64, 2, activation = 'relu', padding = 'same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

#3 compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras42/kaggle_cat_dog/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k42_', date, "_", filename])
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

model.fit(
    x_train,
    y_train,
    validation_split = 0.1,
    callbacks = [es, mcp],
    epochs = 512,
    batch_size = 64,
    verbose = 1
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0, batch_size = 16)

y_pred = model.predict(x_test, batch_size = 16)

print("lead split time :", lead_time_split)
print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))
print("fit time", round(end_time - start_time, 2), "sec")

y_submit = model.predict(xy_test.next(), batch_size = 16)

sample_submission_csv['label'] = y_submit

sample_submission_csv.to_csv(PATH_SUBMISSION + "sampleSubmission_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".csv")

# loss : [0.33697962760925293, 0.8543000221252441]
# acc : 0.8543
# fit time 721.47 sec

# loss : [0.49703797698020935, 0.7527247071266174]
# acc : 0.7527247275272473
# fit time 4320.38 sec

# loss : [0.48730942606925964, 0.7575333118438721]
# acc : 0.7575333333333333
# fit time 1602.58 sec

# loss : [0.5586273074150085, 0.7063000202178955]
# acc : 0.7063
# fit time 1027.27 sec