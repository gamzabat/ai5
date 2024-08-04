import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time

import os

# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

WIDTH = 100
HEIGHT = 100
CHANNEL = 3

#1 data
NP_PATH = "./_data/_save_npy/"

x = np.load(NP_PATH + "keras44_03_x_train.npy")
y = np.load(NP_PATH + "keras44_03_y_train.npy")

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    shuffle = False
)

#2 model
model = Sequential()

model.add(Conv2D(128, 3, input_shape = (100, 100, 3), activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Conv2D(128, 3, activation = 'relu', padding = 'same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Conv2D(128, 3, activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(64, 2, activation = 'relu', padding = 'same'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Conv2D(64, 2, activation = 'relu', padding = 'same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(3, activation = 'softmax'))

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras44/load_03_rps/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k44_', date, "_", filename])
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
    validation_split = 0.25,
    callbacks = [es, mcp],
    epochs = 2560,
    batch_size = 16,
    verbose = 1
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))
print("fit time", round(end_time - start_time, 2), "초")

# loss : [0.1420516073703766, 0.9666666388511658]
# acc : 0.9666666666666667
# fit time 40.22 초