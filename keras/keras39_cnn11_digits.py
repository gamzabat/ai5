import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time

#1 data
x, y = load_digits(return_X_y = True)

print(x.shape)
print(y.shape)

y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 7777,
    stratify = y
)

print(x_train.shape)
print(x_test.shape)

x_train = x_train.reshape(-1, 8, 8, 1)
x_test = x_test.reshape(-1, 8, 8, 1)

#2 model
model = Sequential()

model.add(Conv2D(128, 3, input_shape = (8, 8, 1), activation = 'relu', padding = 'same'))
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

PATH = './_save/keras39/11-disits/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k39_', date, "_", filename])
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

hist = model.fit(
    x_train,
    y_train,
    validation_split = 0.2,
    callbacks = [es, mcp],
    epochs = 2560,
    batch_size = 512,
    verbose = 1
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print("loss :", loss)
print("acc :", accuracy_score(y_test.idxmax(axis = 1), y_pred.argmax(axis = 1)))
print("fit time", round(end_time - start_time, 2), "sec")

# loss : [0.04041828587651253, 0.9916666746139526]
# acc : 0.9916666666666667
# fit time 11.93 초