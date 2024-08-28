import pandas as pd
import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score

import time
import matplotlib.pyplot as plt

import random as rn

from tensorflow.keras.optimizers import Adam
import tensorflow as tf

rn.seed(337)
tf.random.set_seed(337)
np.random.seed(337)

#1 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# np.random.seed(7777)

# class_names = ['apple', 'beaver', 'bottle', 'butterfly', 'castle', 'clock', 'couch', 'leopard', 'rose', 'shark']

# random_idx = np.random.randint(50000, size = 25)

# plt.figure(figsize=(5, 5))

# for i, idx in enumerate(random_idx):
#     plt.subplot(5, 5, i + 1)

#     plt.xticks([])
#     plt.yticks([])

#     plt.imshow(x_train[idx], cmap='gray')

#     plt.xlabel(class_names[y_train[idx][0]])

# plt.show()

print(x_train.shape, y_train.shape) # (50000, 32, 32) (50000,)
print(x_test.shape, y_test.shape)   # (10000, 32, 32) (10000,)

x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

lr = [0.1, 0.01, 0.005, 0.001, 0.0005, 0.0001]

for learning_rate in lr:
    #2 model
    model = Sequential()

    model.add(Conv2D(128, (3, 3), input_shape = (32, 32, 3), activation = 'relu', padding = 'same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (2, 2), activation = 'relu', padding = 'same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (2, 2), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation = 'softmax'))

    model.summary()

    #3 compile
    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(learning_rate = learning_rate), metrics = ['accuracy'])

    ##################### mcp 세이브 파일명 만들기 시작 ###################
    import datetime

    date = datetime.datetime.now()

    print(date) # 2024-07-26 16:49:36.336699
    print(type(date)) # <class 'datetime.datetime'>

    date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

    PATH = './_save/kears67/cifar10/'

    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

    filepath = ''.join([PATH, 'k67_', date, "_", filename])
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
        patience = 5,
        restore_best_weights = True
    )

    rlr = ReduceLROnPlateau(
        monitor = 'val_loss',
        mode = 'auto',
        patience = 5,
        verbose = 1,
        factor = 0.8
    )

    start_time = time.time()

    model.fit(
        x_train,
        y_train,
        validation_split = 0.2,
        callbacks = [es, mcp, rlr],
        epochs = 2560,
        batch_size = 128,
        verbose = 1
    )

    end_time = time.time()

    #4 predict
    loss = model.evaluate(x_test, y_test, verbose = 0)

    y_pred = model.predict(x_test)

    print(y_pred)

    print(y_pred.shape)

    print("rl: {0}, loss :{1}".format(learning_rate, loss))
    print("acc :", accuracy_score(y_test.argmax(axis = 1), y_pred.argmax(axis = 1)))
    print("fit time", round(end_time - start_time, 2), "초")

# lr: 0.1, loss :[2.307277202606201, 0.10000000149011612]
# acc : 0.1
# rl: 0.1, loss :[2.3072760105133057, 0.10000000149011612]
# acc : 0.1

# lr: 0.01, loss :[1.24530827999115, 0.5579000115394592]
# acc : 0.5579
# rl: 0.01, loss :[1.3054596185684204, 0.5260999798774719]
# acc : 0.5261

# lr: 0.005, loss :[0.9508119821548462, 0.670799970626831]
# acc : 0.6708
# rl: 0.005, loss :[0.9460048675537109, 0.675000011920929]
# acc : 0.675

# lr: 0.001, loss :[0.7292011976242065, 0.7472000122070312]
# acc : 0.7472
# rl: 0.001, loss :[0.7247474789619446, 0.7554000020027161]
# acc : 0.7554

# lr: 0.0005, loss :[0.740263819694519, 0.7576000094413757]
# acc : 0.7576
# rl: 0.0005, loss :[0.7405683994293213, 0.7480999827384949]
# acc : 0.7481

# lr: 0.0001, loss :[0.816382110118866, 0.7267000079154968]
# acc : 0.7267
# rl: 0.0001, loss :[0.8270292282104492, 0.7225000262260437]
# acc : 0.7225