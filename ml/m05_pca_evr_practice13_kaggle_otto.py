# https://www.kaggle.com/competitions/otto-group-product-classification-challenge

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

#1 data
PATH = "./_data/kaggle/otto/" # 절대경로

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)
test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)
sample_submission_csv = pd.read_csv(PATH + "sampleSubmission.csv", index_col = 0)

x = train_csv.drop('target', axis = 1)
#x = train_csv

le = LabelEncoder()

train_csv['target'] = le.fit_transform(train_csv['target'])

print(train_csv['target'])

y = pd.get_dummies(train_csv['target'])
#y = train_csv['target']

print(y)

print(train_csv.isna().sum())
print(test_csv.isna().sum())

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    random_state = 7777,
    stratify = y
)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)

x_train = min_max_scaler.transform(x_train)
x_test = min_max_scaler.transform(x_test)

n_components = np.array([61, 81, 91, 93])

for i in reversed(n_components):
    pca = PCA(n_components = i)

    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    #3 model
    model = Sequential()

    model.add(Dense(128, input_dim = i, activation = 'relu'))

    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(9, activation = 'softmax'))

    #3 compile
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    ##################### mcp 세이브 파일명 만들기 시작 ###################
    import datetime

    date = datetime.datetime.now()

    print(date) # 2024-07-26 16:49:36.336699
    print(type(date)) # <class 'datetime.datetime'>

    date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

    PATH = './_save/ml05/13-kaggle-otto/'

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
        monitor = 'val_loss',
        mode = 'min',
        patience = 32,
        restore_best_weights = True
    )

    model.fit(
        x_train,
        y_train,
        validation_split = 0.25,
        callbacks = [es, mcp],
        epochs = 5120,
        batch_size = 256,
        verbose = 2
    )

    #4 predict
    loss = model.evaluate(x_test, y_test, verbose = 0)

    y_pred = model.predict(x_test)

    # print(np.round(y_pred[:10]))

    print("n_components =", i)
    print("loss :", loss)
    print("acc :", accuracy_score(y_test, np.round(y_pred)))

    # y_submit = model.predict(test_csv)

    # print(y_submit[:10])

    # for i in range(9):
    #     sample_submission_csv['Class_' + str(i + 1)] = y_submit[:, i]

    # sample_submission_csv.to_csv(PATH + "sampleSubmission_0729.csv")

# before dropout
# loss : [0.5492791533470154, 0.7920976281166077]
# acc : 0.743939883645766

# after dropout
# loss : [0.5500587821006775, 0.7950872778892517]
# acc : 0.7581609566903684

# n_components = 93
# loss : [0.5492919683456421, 0.7881383299827576]
# acc : 0.7520200387847447

# n_components = 91
# loss : [0.5534992814064026, 0.7937944531440735]
# acc : 0.7402230122818358

# n_components = 81
# loss : [0.562379777431488, 0.7866839170455933]
# acc : 0.732950872656755

# n_components = 61
# loss : [0.567757248878479, 0.7865223288536072]
# acc : 0.7342436974789915