import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time

#1 data
PATH = "./_data/dacon/diabetes/" # 상대경로

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(PATH + "sample_submission.csv", index_col = 0)

train_csv = train_csv[train_csv['BloodPressure'] != 0]
train_csv = train_csv[train_csv['BMI'] > 0.0]
train_csv = train_csv[train_csv['Glucose'] > 0]

x = train_csv.drop(['Outcome'], axis = 1)

y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.70,
    random_state = 7777
)

x_train = x_train.values.reshape(-1, 4, 2)
x_test = x_test.values.reshape(-1, 4, 2)

#2 model
model = Sequential()

# model.add(Conv2D(64, 2, input_shape = (2, 2, 2), activation = 'relu', padding = 'same'))
# model.add(Conv2D(64, 2, activation = 'relu', padding = 'same'))
# model.add(Conv2D(64, 2, activation = 'relu', padding = 'same'))

model.add(LSTM(units = 64, input_shape = (4, 2), return_sequences = True, activation = 'relu'))
model.add(LSTM(units = 64, return_sequences = True, activation = 'relu'))
model.add(LSTM(units = 64, activation = 'relu'))

model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1))

model.summary()

#3 compile
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras59/04-dacon-ddarung/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k59_', date, "_", filename])
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

hist = model.fit(
    x_train,
    y_train,
    validation_split = 0.2,
    callbacks = [es, mcp],
    epochs = 2560,
    batch_size = 256,
    verbose = 1
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(y_test)
print(y_pred)

print("loss :", loss)
print("rs :", r2_score(y_test, y_pred))
print("fit time", round(end_time - start_time, 2), "sec")

# loss : [0.17503024637699127, 0.7675675749778748] - CNN
# rs : 0.14373783099051718
# fit time 3.68 sec

# loss : [0.19247688353061676, 0.745945930480957] - LSTM
# rs : 0.05838742305179556
# fit time 3.74 sec