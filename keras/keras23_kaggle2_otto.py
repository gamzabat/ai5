# https://www.kaggle.com/competitions/otto-group-product-classification-challenge

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf

import time

#1 data
PATH = "C:/ai5/_data/kaggle/otto/" # 절대경로

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
    random_state = 4747,
    stratify = y
)

model = Sequential()

model.add(Dense(128, input_dim = 93, activation = 'relu'))

model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))

model.add(Dense(9, activation = 'softmax'))

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

start_time = time.time()

es = EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',
    patience = 256,
    restore_best_weights = True
)

model.fit(
    x_train,
    y_train,
    validation_split = 0.25,
#    callbacks = [es],
    epochs = 5120,
    batch_size = 1024,
    verbose = 2
)

end_time = time.time()

print("fit time :", round(end_time - start_time, 2), "초")

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

# print(np.round(y_pred[:10]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))

y_submit = model.predict(test_csv)

print(y_submit[:10])

print(np.round(y_submit[:10]))

y_submit = np.round(y_submit)

for i in range(9):
    sample_submission_csv['Class_' + str(i + 1)] = y_submit[:, i].astype('int')

sample_submission_csv.to_csv(PATH + "sampleSubmission_0724.csv")