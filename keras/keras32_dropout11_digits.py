import numpy as np
import pandas as pd

from sklearn.datasets import load_digits

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

#1 data
x, y = load_digits(return_X_y = True)

y = pd.get_dummies(y)

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 7777,
    stratify = y
)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)

x_train = min_max_scaler.transform(x_train)
x_test = min_max_scaler.transform(x_test)

#2 model
model = Sequential()

model.add(Dense(128, input_shape = (64, ), activation = 'relu'))

model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax'))

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras32/11-digits/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k32_', date, "_", filename])
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
    patience = 64,
    restore_best_weights = True
)

model.fit(
    x_train,
    y_train,
    validation_split = 0.25,
    callbacks = [es, mcp],
    epochs = 1000,
    batch_size = 4,
    verbose = 1
)

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(np.round(y_pred[:10]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))

# before dropout
# loss : [0.169927179813385, 0.9722222089767456]
# acc : 0.9666666666666667

# after dropout
# loss : [0.2741842269897461, 0.9638888835906982]
# acc : 0.9611111111111111