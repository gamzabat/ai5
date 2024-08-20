from tensorflow.keras.datasets import reuters

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten, Embedding, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words = 30982,
    test_split = 0.2
)

print(x_train)
print(x_train.shape, x_test.shape) # (8982,) (2246,)
print(y_train.shape, y_test.shape) # (8982,) (2246,)
print(y_train)
print(np.unique(y_train))

print(type(x_train)) # <class 'numpy.ndarray'>
print(type(x_train[0])) # <class 'list'>
print(len(x_train[0]), len(x_train[1]))

print(max([len(i) for i in x_train])) # 2376
print(min([len(i) for i in x_train])) # 13
print(np.average([len(i) for i in x_train])) # 145.5398574927633

# maxall = 0

# for i in x_train:
#     maxs = max([k for k in i])

#     if maxs > maxall:
#         maxall = maxs

# print(maxall)

# preprocessing
x_train = pad_sequences(x_train, padding = 'pre', maxlen = 2376, truncating = 'pre')
x_test = pad_sequences(x_test, padding = 'pre', maxlen = 2376, truncating = 'pre')

y_train = to_categorical(y_train, num_classes = 46)
y_test = to_categorical(y_test, num_classes = 46)

# model
model = Sequential()

model.add(Embedding(30982, 128)) # OK
model.add(LSTM(64)) # (None, 10)
model.add(Dense(64, activation = 'relu'))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))

model.add(Dense(46, activation = 'softmax'))

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

es = EarlyStopping(
    monitor = 'val_accuracy',
    mode = 'max',
    patience = 10,
    restore_best_weights = True
)

model.fit(
    x_train,
    y_train,
    validation_split = 0.1,
    callbacks = [es],
    epochs = 500,
    batch_size = 64,
    verbose = 1
)

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

# y_pred = model.predict(x_question)

print('loss :', loss)
# print('predict :', np.round(y_pred))
# print('predict :', y_pred)

# loss : [2.2799952030181885, 0.6553873419761658]