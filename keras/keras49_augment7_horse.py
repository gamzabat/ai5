from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import time

NP_PATH = "./_data/_save_npy/"

x = np.load(NP_PATH + "keras44_02_x_train.npy")
y = np.load(NP_PATH + "keras44_02_y_train.npy")

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    shuffle = False
)

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

print(x_train.shape[0])

randidx = np.random.randint(x_train.shape[0], size = augment_size)

print(np.min(randidx), np.max(randidx))
print(x_train[0].shape) # (100, 100)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

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

print(x_augmented.shape)

x_train = x_train.reshape(-1, 100, 100, 3)
x_test = x_test.reshape(-1, 100, 100, 3)

print(x_train.shape, x_test.shape)

x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

print(x_train.shape, y_train.shape)

print(y_test.shape)

#2 model
model = Sequential()

model.add(Conv2D(32, 3, activation='relu', input_shape=(100, 100, 3), padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, 3, activation='relu', padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(128, 3, activation='relu', padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, 3, activation='relu', padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten()) 
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu')) 
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))

model.summary()

#3 compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras49/horse/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k49_', date, "_", filename])
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
    validation_split = 0.2,
    callbacks = [es, mcp],
    epochs = 2560,
    batch_size = 64,
    verbose = 1
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(y_test.shape)
print(y_pred.shape)

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))
print("fit time", round(end_time - start_time, 2), "sec")

# loss : [0.07312098145484924, 0.9805825352668762]
# acc : 0.9805825242718447
# fit time 34.17 sec