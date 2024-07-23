# 12 dacon copy

# https://www.kaggle.com/competitions/playground-series-s4e1/overview

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

import time

PATH = "./_data/kaggle/playground-series-s4e1/" # 상대경로

#0 replace data
# train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

# print(train_csv['Geography'].value_counts())

# train_csv['Geography'] = train_csv['Geography'].replace('France', value = 1)
# train_csv['Geography'] = train_csv['Geography'].replace('Spain', value = 2)
# train_csv['Geography'] = train_csv['Geography'].replace('Germany', value = 3)

# train_csv['Gender'] = train_csv['Gender'].replace('Male', value = 1)
# train_csv['Gender'] = train_csv['Gender'].replace('Female', value = 2)

# train_csv.to_csv(PATH + "replaced_train.csv")

# test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

# test_csv['Geography'] = test_csv['Geography'].replace('France', value = 1)
# test_csv['Geography'] = test_csv['Geography'].replace('Spain', value = 2)
# test_csv['Geography'] = test_csv['Geography'].replace('Germany', value = 3)

# test_csv['Gender'] = test_csv['Gender'].replace('Male', value = 1)
# test_csv['Gender'] = test_csv['Gender'].replace('Female', value = 2)

# test_csv.to_csv(PATH + "replaced_test.csv")

train_csv = pd.read_csv(PATH + "replaced_train.csv", index_col = 0)

print(train_csv) # [165034 rows x 13 columns]

test_csv = pd.read_csv(PATH + "replaced_test.csv", index_col = 0)

print(test_csv) # [110023 rows x 12 columns]

submission_csv = pd.read_csv(PATH + "sample_submission.csv", index_col = 0)

print(submission_csv) # [110023 rows x 1 columns]

train_csv.info() # CustomerId Surname  CreditScore Geography Gender   Age  Tenure    Balance  NumOfProducts  HasCrCard  IsActiveMember  EstimatedSalary  Exited

train_csv = train_csv.drop(['CustomerId', 'Surname'], axis = 1)

train_csv.info()

test_csv = test_csv.drop(['CustomerId', 'Surname'], axis = 1)

test_csv.info()

###############################################
from sklearn.preprocessing import MinMaxScaler

train_scaler = MinMaxScaler()

train_csv_copy = train_csv.copy()

train_csv_copy = train_csv_copy.drop(['Exited'], axis = 1)

train_scaler.fit(train_csv_copy)

train_csv_scaled = train_scaler.transform(train_csv_copy)

train_csv = pd.concat([pd.DataFrame(data = train_csv_scaled), train_csv['Exited']], axis = 1)

test_scaler = MinMaxScaler()

test_csv_copy = test_csv.copy()

test_scaler.fit(test_csv_copy)

test_csv_scaled = test_scaler.transform(test_csv_copy)

test_csv = pd.DataFrame(data = test_csv_scaled)
###############################################
x = train_csv.drop(['Exited'], axis = 1)

y = train_csv['Exited']

print(y.value_counts())

input()

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.80,
    random_state = 1186
)

#2 model
model = Sequential()

model.add(Dense(32, input_dim = 10, activation = 'relu'))

model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

#3 compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start_time = time.time()

es = EarlyStopping( 
    monitor = 'val_loss',
    mode = 'min',
    patience = 32,
    restore_best_weights = True
)

hist = model.fit(
    x_train,
    y_train,
    validation_split = 0.3,
    callbacks = [es],
    epochs = 256,
    batch_size = 256,
    verbose = 2
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(y_pred[:30])

print(np.round(y_pred[:30]))

print("acc :", accuracy_score(y_test, np.round(y_pred)))

y_submit = model.predict(test_csv)

submission_csv['Exited'] = np.round(y_submit)

submission_csv.to_csv(PATH + "sample_submission_0723.csv")