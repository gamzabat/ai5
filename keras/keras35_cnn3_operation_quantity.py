import pandas as pd
import numpy as np

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score

import time

#1 data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,) 

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#2 model
model = Sequential()

model.add(Conv2D(10, (2, 2), input_shape = (28, 28, 1)))
model.add(Conv2D(filters = 20, kernel_size = (3, 3)))
# input shape - (batch_size, height, width, channels)
# output shape - (batch_size, new_height, new_width, filters)
model.add(Conv2D(15, (4, 4)))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))

model.add(Dense(10, activation = 'softmax'))

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 32,
    restore_best_weights = True
)

start_time = time.time()

model.fit(
    x_train,
    y_train,
    validation_split = 0.25,
    callbacks = [es],
    epochs = 3,
    batch_size = 128,
    verbose = 2
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))
print("fit time", round(end_time - start_time, 2), "초")