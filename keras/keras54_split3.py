import numpy as np

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU, BatchNormalization, Dropout, Flatten

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

a = np.array(range(1, 101)) # 101에서 107까지 찾을것

print(a)

size = 6

def split_x(dataset, size):
    result = []

    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]

        result.append(subset)

    return np.array(result)

x_split = split_x(a, size)

print('---------------------------')
print(x_split)
print('---------------------------')
print(x_split.shape)

x = x_split[:, :-1]
y = x_split[:, -1]

print('---------------------------')
print(x)
print('---------------------------')
print(y)
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 7777
)

#2 model
model = Sequential()

model.add(LSTM(16, input_shape=(5, 1), activation = 'tanh')) # 3 : timesteps, 1 : features
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3 compile, fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras54/split4/'

filename = '{epoch:04d}_{loss:.8f}.hdf5'

filepath = ''.join([PATH, 'k54_', date, "_", filename])
##################### mcp 세이브 파일명 만들기 끝 ###################

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1,
    save_best_only = True,
    filepath = filepath
)

es = EarlyStopping(
    monitor = 'loss',
    mode = 'min',
    patience = 30,
    restore_best_weights = True
)

model.fit(
    x_train,
    y_train,
    validation_split = 0.1,
    callbacks = [mcp],
    epochs = 5000,
    batch_size = 8,
    verbose = 1
)

results = model.evaluate(x, y)

print("loss :", results)

#model = load_model('./_save/k52_lstm_scale2.hdf5')

x_predict = np.array(range(96, 106)).reshape(1, 10, 1)

y_pred = model.predict(x_predict)

print("pred :", y_pred)

# pred : [[80.57921]]