import numpy as np
import pandas as pd

from sklearn.datasets import fetch_covtype

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time

from sklearn.preprocessing import OneHotEncoder

#1 data
dataset = fetch_covtype()

x = pd.DataFrame(dataset.data).iloc[:, :13]

y = pd.get_dummies(dataset.target)

#y = pd.DataFrame(to_categorical(dataset.target))

print(y.head())

# y = OneHotEncoder(sparse = False).fit_transform(dataset.target.reshape(-1, 1))

# print(x)
# print(y.shape)

# print(pd.DataFrame(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.9,
    random_state = 42,
    stratify = y
)

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

# print(pd.DataFrame(y_train).value_counts())

#2 model
model = Sequential()

model.add(Dense(64, input_dim = 13, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))

model.add(Dense(7, activation = 'softmax'))

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 32,
    restore_best_weights = True
)

model.fit(
    x_train,
    y_train,
    validation_split = 0.3,
    callbacks = [es],
    epochs = 1000,
    batch_size = 512,
    verbose = 1
)

end_time = time.time()

print("fit time :", round(end_time - start_time, 2), "초")

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(np.round(y_pred[:10]))

print("acc :", accuracy_score(y_test, np.round(y_pred)))

# acc : 0.8836690963228144
# acc : 0.8947703587687065
# acc : 0.9113283535850745
# acc : 0.925252142783381