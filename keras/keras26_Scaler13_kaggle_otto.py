# https://www.kaggle.com/competitions/otto-group-product-classification-challenge

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

model.add(Dense(128, input_dim = 93, activation = 'relu'))

model.add(Dense(128, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))

model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))

model.add(Dense(9, activation = 'softmax'))

#3 compile
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

start_time = time.time()

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
    callbacks = [es],
    epochs = 5120,
    batch_size = 256,
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

y_submit = np.round(model.predict(test_csv), 1)

print(y_submit[:10])

for i in range(9):
    sample_submission_csv['Class_' + str(i + 1)] = y_submit[:, i]

sample_submission_csv.to_csv(PATH + "sampleSubmission_0725.csv")

# before scaler
# loss : [0.5735519528388977, 0.7898351550102234]
# acc : 0.7520200387847447

# after minMaxScaler
# loss : [0.5461804270744324, 0.7933904528617859]
# acc : 0.7529896574014221

# after standardScaler
# loss : [0.5280560255050659, 0.8011474013328552]
# acc : 0.7657563025210085

# after minAbsScaler
# loss : [0.5290956497192383, 0.7955721020698547]
# acc : 0.7580801551389786

# after robustScaler
# loss : [0.5230527520179749, 0.8013089895248413]
# acc : 0.7622010342598577