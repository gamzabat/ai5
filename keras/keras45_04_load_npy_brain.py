import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time

WIDTH = 200
HEIGHT = 200

#1 data
NP_PATH = "./_data/_save_npy/"

x_train = np.load(NP_PATH + "keras44_01_x_train.npy")
y_train = np.load(NP_PATH + "keras44_01_y_train.npy")
x_test = np.load(NP_PATH + "keras44_01_x_test.npy")
y_test = np.load(NP_PATH + "keras44_01_y_test.npy")

#2 model
model = Sequential()

model.add(Conv2D(32, 3, input_shape = (WIDTH, HEIGHT, 1), activation = 'relu', padding = 'same'))
model.add(Conv2D(32, 3, activation = 'relu', padding = 'same'))
model.add(MaxPooling2D())
model.add(Dropout(0.3))
model.add(Conv2D(32, 2, activation = 'relu', padding = 'same'))
model.add(MaxPooling2D())
model.add(Conv2D(16, 2, activation = 'relu', padding = 'same'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation = 'sigmoid'))

#3 compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras44/load_01_brain/'

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
    batch_size = 32,
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