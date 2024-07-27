# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/overview

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import time

#1 data
PATH = "./_data/kaggle/santander/" # 절대경로

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
    random_state = 7777,
    stratify = y
)

# min_max_scaler = MinMaxScaler()

# min_max_scaler.fit(x_train)

# x_train = min_max_scaler.transform(x_train)
# x_test = min_max_scaler.transform(x_test)
# test_csv = min_max_scaler.transform(test_csv)

# standard_scaler = StandardScaler().fit(x_train)

# x_train = standard_scaler.transform(x_train)
# x_test = standard_scaler.transform(x_test)
# test_csv = standard_scaler.transform(test_csv)

# max_abs_scaler = MaxAbsScaler().fit(x_train)

# x_train = max_abs_scaler.transform(x_train)
# x_test = max_abs_scaler.transform(x_test)
# test_csv = max_abs_scaler.transform(test_csv)

robust_scaler = RobustScaler().fit(x_train)

x_train = robust_scaler.transform(x_train)
x_test = robust_scaler.transform(x_test)
test_csv = robust_scaler.transform(test_csv)

#3 model
model = Sequential()

model.add(Dense(256, input_dim = 200, activation = 'relu'))

model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
model.add(Dense(256, activation = 'relu'))
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
    batch_size = 1024,
    verbose = 1
)

end_time = time.time()

print("fit time :", round(end_time - start_time, 2), "초")

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(np.round(y_pred[:10]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))

y_submit = model.predict(test_csv)

print(y_submit[:10])

# print(np.round(y_submit[:, 1]))

sample_submission_csv['target'] = np.round(y_submit).astype("int")

sample_submission_csv.to_csv(PATH + "sample_submission_0725.csv")

# before scaler
# loss : [0.24630630016326904, 0.9092000126838684]
# acc : 0.9092

# after minMaxScaler
# loss : [0.23003046214580536, 0.9156000018119812]
# acc : 0.9156

# after standardScaler
# loss : [0.24239280819892883, 0.9444444179534912]
# acc : 0.9388888888888889

# after minAbsScaler
# loss : [0.22922451794147491, 0.916700005531311]
# acc : 0.9167

# after robustScaler
# loss : [0.23174616694450378, 0.9117000102996826]
# acc : 0.9117