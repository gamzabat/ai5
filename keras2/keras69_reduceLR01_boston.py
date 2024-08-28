from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import numpy as np

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import random as rn

from tensorflow.keras.optimizers import Adam
import tensorflow as tf

rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)

# data
datasets = load_boston()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.75,
    random_state = 337
)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# model
model = Sequential()

model.add(Dense(10, input_dim = 13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 100,
    verbose = 1,
    restore_best_weights = True
)

rlr = ReduceLROnPlateau(
    monitor = 'val_loss',
    mode = 'auto',
    patience = 25,
    verbose = 1,
    factor = 0.8
)

# train
from tensorflow.keras.optimizers import Adam

learning_rate = 0.01

model.compile(loss = 'mse', optimizer = Adam(learning_rate = learning_rate))

model.fit(
    x_train,
    y_train,
    validation_split = 0.1,
    epochs = 1000,
    batch_size = 32,
    callbacks = [es, rlr]
)

loss = model.evaluate(x_test, y_test, verbose = 0)

print("lr: {0}, loss :{1}".format(learning_rate, loss))

y_predict = model.predict(x_test, verbose = 0)

r2 = r2_score(y_test, y_predict)

print("r2 : {0}".format(r2))

########################## practice ##########################