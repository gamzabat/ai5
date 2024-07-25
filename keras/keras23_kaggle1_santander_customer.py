# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time

#1 data
PATH = "C:/ai5/_data/kaggle/santander/" # 절대경로

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)
test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)
sample_submission_csv = pd.read_csv(PATH + "sample_submission.csv", index_col = 0)

x = train_csv.drop('target', axis = 1)

#y = pd.get_dummies(train_csv['target'])
y = train_csv['target']

print(y)

# print(train_csv.isna().sum())
# print(test_csv.isna().sum())

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.95,
    random_state = 6666,
    stratify = y
)

model = Sequential()

model.add(Dense(256, input_dim = 200, activation = 'relu'))

model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

#3 compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 16,
    restore_best_weights = True
)

model.fit(
    x_train,
    y_train,
    validation_split = 0.1,
    callbacks = [es],
    epochs = 1000,
    batch_size = 8196,
    verbose = 1
)

end_time = time.time()

print("fit time :", round(end_time - start_time, 2), "초")

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(np.round(y_pred[:10]))

print("acc :", accuracy_score(y_test, np.round(y_pred)))

y_submit = model.predict(test_csv)

print(y_submit[:10])

print(np.round(y_submit[:, 1]))

sample_submission_csv['target'] = np.round(y_submit)

sample_submission_csv.to_csv(PATH + "sample_submission_0724.csv")