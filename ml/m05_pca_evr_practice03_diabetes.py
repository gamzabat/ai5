from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes

from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np

from sklearn.decomposition import PCA

#1 data
datasets = load_diabetes()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.70,
    random_state = 7777
)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)

x_train = min_max_scaler.transform(x_train)
x_test = min_max_scaler.transform(x_test)

n_components = np.array([7, 8, 9, 10])

for i in reversed(n_components):
    pca = PCA(n_components = i)

    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    #2 model
    model = Sequential()

    model.add(Dense(32, input_dim = i, activation = 'relu'))

    model.add(Dropout(0.3))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation = 'relu'))

    model.add(Dense(1))

    #3 compile
    model.compile(loss = 'mse', optimizer = 'adam')

    ##################### mcp 세이브 파일명 만들기 시작 ###################
    import datetime

    date = datetime.datetime.now()

    print(date) # 2024-07-26 16:49:36.336699
    print(type(date)) # <class 'datetime.datetime'>

    date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

    PATH = './_save/ml05/03-diabetes/'

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

    es = EarlyStopping(
        monitor = "val_loss",
        mode = "min",
        patience = 64,
        restore_best_weights = True
    )

    hist = model.fit(x_train,
                    y_train,
                    validation_split = 0.2,
                    callbacks = [es, mcp],
                    epochs = 1000,
                    batch_size = 32)

    #4 predict
    loss = model.evaluate(x_test, y_test)

    y_predict = model.predict(x_test)

    r2 = r2_score(y_test, y_predict)

    print("n_components =", i)
    print("loss :", loss)
    print("rs :", r2)

# loss : 2436.35791015625
# rs : 0.5897085199507599

# n_components = 10
# loss : 3381.647705078125
# rs : 0.4305182589262825

# n_components = 9
# loss : 4197.640625
# rs : 0.2931021985545549

# n_components = 8
# loss : 3356.588623046875
# rs : 0.4347383009436032

# n_components = 7
# loss : 3895.359130859375
# rs : 0.3440074202311516