from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score

import time

#1 data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32) (50000,)
print(x_test.shape, y_test.shape)   # (10000, 32, 32) (10000,)

x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2 model
model = Sequential()

model.add(Conv2D(100, (3, 3), input_shape = (32, 32, 3), activation = 'relu'))
model.add(Conv2D(50, (3, 3), activation = 'relu'))
model.add(Conv2D(20, (2, 2), activation = 'relu'))
model.add(Flatten())
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(100, activation = 'softmax'))

model.summary()

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 100,
    restore_best_weights = True
)

model.fit(
    x_train,
    y_train,
    validation_split = 0.2,
    callbacks = [es],
    epochs = 2560,
    batch_size = 100,
    verbose = 1
)

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print("loss :", loss)
print("acc :", accuracy_score(y_test.argmax(axis = 1), y_pred.argmax(axis = 1)))