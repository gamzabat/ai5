from tensorflow.keras.datasets import imdb

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import StandardScaler, MinMaxScaler

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words = 30981
)

print(x_train)
print(x_train.shape, x_test.shape) # (25000,) (25000,)
print(y_train.shape, y_test.shape) # (25000,) (25000,)
print(y_train)
print(np.unique(y_train))

print(type(x_train)) # <class 'numpy.ndarray'>
print(type(x_train[0])) # <class 'list'>
print(len(x_train[0]), len(x_train[1])) # 218 189

print(max([len(i) for i in x_train])) # 2494
print(min([len(i) for i in x_train])) # 11
print(np.average([len(i) for i in x_train])) # 238.71364

# maxall = 0

# for i in x_train:
#     maxs = max([k for k in i])

#     if maxs > maxall:
#         maxall = maxs

# print(maxall)

# preprocessing
x_train = pad_sequences(x_train, padding = 'pre', maxlen = 2494, truncating = 'pre')
x_test = pad_sequences(x_test, padding = 'pre', maxlen = 2494, truncating = 'pre')

# model
model = Sequential()

model.add(Embedding(30981, 128)) # OK
model.add(LSTM(64)) # (None, 10)
model.add(Dense(64, activation = 'relu'))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 10,
    restore_best_weights = True
)

model.fit(
    x_train,
    y_train,
    validation_split = 0.1,
    callbacks = [es],
    epochs = 500,
    batch_size = 512,
    verbose = 1
)

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

# y_pred = model.predict(x_question)

print('loss :', loss)
# print('predict :', np.round(y_pred))
# print('predict :', y_pred)

# loss : [0.30942854285240173, 0.8744000196456909]