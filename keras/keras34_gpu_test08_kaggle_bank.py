# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import tensorflow as tf

import numpy as np
import pandas as pd

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from keras.callbacks import EarlyStopping, ModelCheckpoint

import time

gpus = tf.config.experimental.list_physical_devices('GPU')

print(gpus)

PATH = "./_data/kaggle/playground-series-s4e1/" # 상대경로

le = LabelEncoder()

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

le.fit(train_csv['Geography'])

train_csv['Geography'] = le.fit_transform(train_csv['Geography'])
test_csv['Geography'] = le.fit_transform(test_csv['Geography'])

le.fit(train_csv['Gender'])

train_csv['Gender'] = le.fit_transform(train_csv['Gender'])
test_csv['Gender'] = le.fit_transform(test_csv['Gender'])

submission_csv = pd.read_csv(PATH + "sample_submission.csv", index_col = 0)

print(submission_csv) # [110023 rows x 1 columns]

train_csv.info() # CustomerId Surname  CreditScore Geography Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited

train_csv = train_csv.drop(['CustomerId', 'Surname'], axis = 1)

train_csv.info()

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis = 1)

test_csv.info()

x = train_csv.drop(['Exited'], axis = 1)

y = train_csv['Exited']

print(y.value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.80,
    random_state = 7777
)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)

x_train = min_max_scaler.transform(x_train)
x_test = min_max_scaler.transform(x_test)

#2 model
# model = Sequential()

# model.add(Dense(16, input_dim = 10, activation = 'relu'))

# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))

# model.add(Dense(1, activation = 'sigmoid'))

input1 = Input(shape = (10,))

dense1 = Dense(16, activation = 'relu')(input1)
dense2 = Dense(16, activation = 'relu')(dense1)
dense3 = Dense(16, activation = 'relu')(dense2)
dense4 = Dense(16, activation = 'relu')(dense3)
dense5 = Dense(16, activation = 'relu')(dense4)
drop1 = Dropout(0.2)(dense5)
dense6 = Dense(16, activation = 'relu')(drop1)
dense7 = Dense(16, activation = 'relu')(dense6)
dense8 = Dense(16, activation = 'relu')(dense7)
dense9 = Dense(16, activation = 'relu')(dense8)

output1 = Dense(1, activation = 'sigmoid')(dense9)

model = Model(inputs = input1, outputs = output1)

#3 compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras32/08-kaggle-bank/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k32_', date, "_", filename])
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
    patience = 128,
    restore_best_weights = True
)

start_time = time.time()

hist = model.fit(
    x_train,
    y_train,
    validation_split = 0.3,
    callbacks = [es, mcp],
    epochs = 1024,
    batch_size = 512,
    verbose = 2
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(y_pred[:30])

print(np.round(y_pred[:30]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))
print("fit time", "gpu on" if (len(gpus) > 0) else "gpu off", round(end_time - start_time, 2), "초")

y_submit = model.predict(test_csv)

submission_csv['Exited'] = np.round(y_submit, 1)

submission_csv.to_csv(PATH + "sample_submission_0729.csv")

# fit time gpu on 261.03 초
# fit time gpu off 34.58 초

# before dropout
# loss : [0.3230242431163788, 0.864907443523407]
# acc : 0.8649074438755415

# after dropout
# loss : [0.3342232406139374, 0.8612112402915955]
# acc : 0.8612112582179537