# 12 dacon copy

# https://dacon.io/competitions/official/236068/overview/description

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

import time

PATH = "./_data/dacon/diabetes/" # 상대경로

#1 data
train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(PATH + "sample_submission.csv", index_col = 0)

print(submission_csv) # [715 rows x 1 columns] - 결측치

print(test_csv.isna().sum())

train_csv.info() # Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome

test_csv.info()

print(train_csv.shape)

train_csv = train_csv[train_csv['BloodPressure'] != 0]
train_csv = train_csv[train_csv['BMI'] > 0.0]
train_csv = train_csv[train_csv['Glucose'] > 0]
 
print(train_csv['Glucose'].value_counts())

print(train_csv.shape)

train_csv.info()

x = train_csv.drop(['Outcome'], axis = 1)

y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.80,
    random_state = 3333
)

# min_max_scaler = MinMaxScaler()

# min_max_scaler.fit(x_train)

# x_train = min_max_scaler.transform(x_train)
# x_test = min_max_scaler.transform(x_test)
# test_csv = min_max_scaler.transform(test_csv)

# stardard_scaler = StandardScaler().fit(x_train)

# x_train = stardard_scaler.transform(x_train)
# x_test = stardard_scaler.transform(x_test)
# test_csv = stardard_scaler.transform(test_csv)

# max_abs_scaler = MaxAbsScaler().fit(x_train)

# x_train = max_abs_scaler.transform(x_train)
# x_test = max_abs_scaler.transform(x_test)
# test_csv = max_abs_scaler.transform(test_csv)

robust_scaler = RobustScaler().fit(x_train)

x_train = robust_scaler.transform(x_train)
x_test = robust_scaler.transform(x_test)
test_csv = robust_scaler.transform(test_csv)

#2 model
model = Sequential()

model.add(Dense(16, input_dim = 8, activation = 'relu'))

model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
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
    patience = 50,
    restore_best_weights = True
)

hist = model.fit(
    x_train,
    y_train,
    validation_split = 0.3,
    callbacks = [es],
    epochs = 1000,
    batch_size = 4,
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

y_submit = model.predict(test_csv)

submission_csv['Outcome'] = np.round(y_submit)

submission_csv.to_csv(PATH + "sample_submission_0725.csv")

# before scaler
# loss : [0.5295171141624451, 0.7177419066429138]
# acc : 0.717741935483871

# after minMaxScaler
# loss : [0.48980429768562317, 0.75]
# acc : 0.75

# after standardScaler
# loss : [0.46480658650398254, 0.7580645084381104]
# acc : 0.7580645161290323

# after minAbsScaler
# loss : [0.5224670767784119, 0.725806474685669]
# acc : 0.7258064516129032

# after robustScaler
# loss : [0.5278043150901794, 0.75]
# acc : 0.75