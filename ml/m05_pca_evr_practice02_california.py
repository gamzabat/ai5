from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.datasets import fetch_california_housing

from sklearn.decomposition import PCA

import numpy as np

import time

#1 data
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.7,
    random_state = 7755
)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)

x_train = min_max_scaler.transform(x_train)
x_test = min_max_scaler.transform(x_test)

n_components = np.array([3, 4, 6, 8])

for i in reversed(n_components):
    pca = PCA(n_components = i)

    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    # model
    model = Sequential()

    model.add(Dense(16, input_dim = x_train.shape[1], activation = 'relu'))

    model.add(Dropout(0.3))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation = 'relu'))

    model.add(Dense(1))

    # compile
    model.compile(loss = 'mse', optimizer = 'adam')

    ##################### mcp 세이브 파일명 만들기 시작 ###################
    import datetime

    date = datetime.datetime.now()

    date = date.strftime("%y%m%d_%H%M%S")

    PATH = './_save/ml05/02-california/'

    filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

    filepath = ''.join([PATH, 'ml05_', date, "_", filename])
    ##################### mcp 세이브 파일명 만들기 끝 ###################

    mcp = ModelCheckpoint(
        monitor = 'val_loss',
        mode = 'auto',
        verbose = 1,
        save_best_only = True,
        filepath = filepath
    )

    start_time = time.time()

    es = EarlyStopping(
        monitor = "val_loss",
        mode = "min",
        patience = 10,
        restore_best_weights = True
    )

    hist = model.fit(x_train,
                    y_train,
                    validation_split = 0.2,
                    callbacks = [es, mcp],
                    epochs = 1000,
                    batch_size = 32)

    end_time = time.time()

    # predict
    loss = model.evaluate(x_test, y_test)

    y_predict = model.predict(x_test)

    r2 = r2_score(y_test, y_predict)

    print("loss :", loss)
    print("rs :", r2)
    print("fit time", round(end_time - start_time, 2), "초")

# loss : 0.6507984399795532
# rs : 0.5206967566448091
# fit time 11.56 초

# n_components = 8
# loss : 0.7086731195449829
# rs : 0.47807284887408796

# n_components = 6
# loss : 0.5678874254226685
# rs : 0.5817594644987529

# n_components = 4
# loss : 0.5330760478973389
# rs : 0.60739756869247

# n_components = 3
# loss : 0.8097451329231262
# rs : 0.4036349956649311