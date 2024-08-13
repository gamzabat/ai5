import pandas as pd
import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, LSTM, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score

import time
import matplotlib.pyplot as plt

#1 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# np.random.seed(7777)

# class_names = ['apple', 'beaver', 'bottle', 'butterfly', 'castle', 'clock', 'couch', 'leopard', 'rose', 'shark']

# random_idx = np.random.randint(50000, size = 25)

# plt.figure(figsize=(5, 5))

# for i, idx in enumerate(random_idx):
#     plt.subplot(5, 5, i + 1)

#     plt.xticks([])
#     plt.yticks([])

#     plt.imshow(x_train[idx], cmap='gray')

#     plt.xlabel(class_names[y_train[idx][0]])

# plt.show()

print(x_train.shape, y_train.shape) # (50000, 32, 32) (50000,)
print(x_test.shape, y_test.shape)   # (10000, 32, 32) (10000,)

x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(-1, 96, 32)
x_test = x_test.reshape(-1, 96, 32)

#2 model
model = Sequential()

model.add(Conv1D(filters = 10, kernel_size = 3, input_shape = (96, 32), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))

model.add(Dense(10, activation = 'softmax'))

model.summary()

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras60/cifar10/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k60_', date, "_", filename])
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

model.fit(
    x_train,
    y_train,
    validation_split = 0.2,
    callbacks = [es, mcp],
    epochs = 2560,
    batch_size = 2048,
    verbose = 1
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(y_pred)

print(y_pred.shape)

print("loss :", loss)
print("acc :", accuracy_score(y_test.argmax(axis = 1), y_pred.argmax(axis = 1)))
print("fit time", round(end_time - start_time, 2), "초")

# loss : [0.9279019236564636, 0.6715999841690063] - CNN
# acc : 0.6716
# fit time 133.18 초

# loss : [2.3018174171447754, 0.11159999668598175] - LSTM
# acc : 0.1116
# fit time 1108.76 초

# loss : [1.4231764078140259, 0.4973999857902527] - Conv1D
# acc : 0.4974
# fit time 15.61 초