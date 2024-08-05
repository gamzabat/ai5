import pandas as pd
import numpy as np

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

#1 data
PATH_TRAIN = "./_data/image/cat_and_dog/train/"
PATH_SUBMIT = "./_data/image/cat_and_dog/test/"

train_image_datagen = ImageDataGenerator(
    rescale = 1. / 255,           # 데이터 스케일링
)

xy_image_train = train_image_datagen.flow_from_directory(
    PATH_TRAIN,
    target_size = (100, 100),
    batch_size = 20000,
    class_mode = 'binary',
    color_mode = 'rgb',
    shuffle = True
)

print(xy_image_train[0][0].shape, xy_image_train[0][1].shape)

NP_PATH = "./_data/_save_npy/"

PATH_TRAIN = "./_data/kaggle/cat_dog/train"
PATH_SUBMIT = "./_data/kaggle/cat_dog/test"

PATH_SUBMISSION = "./_data/kaggle/cat_dog/"

sample_submission_csv = pd.read_csv(PATH_SUBMISSION + "sample_submission.csv", index_col = 0)

x = np.load(NP_PATH + "keras42_good_x_train.npy")
y = np.load(NP_PATH + "keras42_good_y_train.npy")
xy_test = np.load(NP_PATH + "keras42_good_x_test.npy")

print(type(x))

print(x.shape, y.shape)

x = np.concatenate((x, xy_image_train[0][0]))
y = np.concatenate((y, xy_image_train[0][1]))

train_datagen = ImageDataGenerator(
    rescale = 1.0 / 255,
    horizontal_flip = True,        # 수평 뒤집기 = 증폭(완전 다른 데이터가 하나 더 생겼다)
    vertical_flip = True,         # 수직 뒤집기 = 증폭
    # width_shift_range = 0.1,      # 평행 이동   = 증폭
    # height_shift_range = 0.1,     # 평행 이동 수직
    rotation_range = 15,           # 정해진 각도만큼 이미지 회전
    # zoom_range = 1.2,             # 축소 또는 확대
    # shear_range = 0.7,            # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환.
    fill_mode = 'nearest',        # 원래 있던 가까운 놈으로 채운다.
)

augment_size = 5000

randidx = np.random.randint(x.shape[0], size = augment_size)

print(np.min(randidx), np.max(randidx))
print(x[0].shape) # (100, 100)

x_augmented = x[randidx].copy()
y_augmented = y[randidx].copy()

print(x_augmented.shape, y_augmented.shape)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 3)

print(x_augmented.shape)

x_augmented = train_datagen.flow(
    x_augmented,
    y_augmented,
    batch_size = augment_size,
    shuffle = False
).next()[0]

x = np.concatenate((x, x_augmented))
y = np.concatenate((y, y_augmented))

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.9,
    stratify = y,
    random_state = 11
)

#2 model
model = Sequential()

# model.add(Conv2D(32, 3, input_shape = (100, 100, 3), activation = 'relu', padding = 'same'))
# model.add(Conv2D(32, 3, activation = 'relu', padding = 'same'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.3))
# model.add(Conv2D(32, 2, activation = 'relu', padding = 'same'))
# model.add(MaxPooling2D())
# model.add(Conv2D(16, 2, activation = 'relu', padding = 'same'))
# model.add(Flatten())
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(16, activation = 'relu'))

# model.add(Dense(1, activation = 'sigmoid'))

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3), padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=128, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu', padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten()) 
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu')) 
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))

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
    patience = 30,
    restore_best_weights = True
)

start_time = time.time()

model.fit(
    x_train,
    y_train,
    validation_split = 0.1,
    callbacks = [es, mcp],
    epochs = 1024,
    batch_size = 16,
    verbose = 1
)

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0, batch_size = 16)

print("end of evaluation")

y_pred = model.predict(x_test, batch_size = 16)

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))
print("fit time", round(time.time() - start_time, 2), "sec")

y_submit = model.predict(xy_test, batch_size = 16)
y_submit = np.clip(y_submit, 1e-6, 1-(1e-6))

sample_submission_csv['label'] = y_submit

sample_submission_csv.to_csv(PATH_SUBMISSION + "sampleSubmission_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".csv")

# loss : [0.33697962760925293, 0.8543000221252441] -> 0.30366
# acc : 0.8543
# fit time 721.47 sec

# loss : [0.46252983808517456, 0.7623000144958496] -> 0.29466
# acc : 0.7623