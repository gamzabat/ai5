import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time

#1 data
dataset = load_iris()

# print(dataset.DESCR)
# print(dataset.feature_names)

x = dataset.data

#y = to_categorical(dataset.target)

print(dataset.target)
print(dataset.target.shape)

y = OneHotEncoder(sparse = False).fit_transform(dataset.target.reshape(-1, 1))

#y = pd.get_dummies(dataset.target)

print(x.shape, y.shape)

print(y)
print(np.unique(y, return_counts = True))
# print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.7,
    random_state = 7777
)

print(x_train.shape, y_train.shape)

print(x_test.shape, y_test.shape)

model = Sequential()

model.add(Dense(16, input_dim = 4, activation = 'relu'))

model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))

model.add(Dense(3, activation = 'softmax'))

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 30,
    restore_best_weights = True
)

hist = model.fit(
    x_train,
    y_train,
    validation_split = 0.3,
    callbacks = [es],
    epochs = 1000,
    batch_size = 8,
    verbose = 2
)

end_time = time.time()

loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(np.round(y_pred[:30]))

print("acc :", accuracy_score(y_test, np.round(y_pred)))