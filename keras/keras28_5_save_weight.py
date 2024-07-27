import sklearn as sk

print(sk.__version__) # 0.24.2

import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from sklearn.datasets import load_boston

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import time

# data
dataset = load_boston()

print(dataset)
print(dataset.DESCR)
print(dataset.feature_names) # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

x = dataset.data
y = dataset.target

print(x)
print(x.shape) # (506, 13)

print(y)
print(y.shape) # (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.7,
    random_state = 7777
)

# min_max_scaler = MinMaxScaler()

# min_max_scaler.fit(x_train)

# x_train = min_max_scaler.transform(x_train)
# x_test = min_max_scaler.transform(x_test)

# stardard_scaler = StandardScaler().fit(x_train)

# x_train = stardard_scaler.transform(x_train)
# x_test = stardard_scaler.transform(x_test)

# max_abs_scaler = MaxAbsScaler().fit(x_train)

# x_train = max_abs_scaler.transform(x_train)
# x_test = max_abs_scaler.transform(x_test)

robust_scaler = RobustScaler().fit(x_train)

x_train = robust_scaler.transform(x_train)
x_test = robust_scaler.transform(x_test)

print(np.min(x_train), np.max(x_train))
print(np.min(x_test), np.max(x_test))

# model
model = Sequential()

model.add(Dense(10, input_dim = 13, activation = 'relu'))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1))

model.summary()

# model.save("./_save/keras28/keras28_1_save_model.h5")

model.save_weights("./_save/keras28/keras28_5_save_model_weights1.h5")

# compile
model.compile(loss = 'mse', optimizer = 'adam')

start_time = time.time()

hist = model.fit(x_train,
          y_train,
          validation_split = 0.2,
          epochs = 333,
          batch_size = 32,
          verbose = 1)

model.save("./_save/keras28/keras28_3_save_model.h5")

model.save_weights("./_save/keras28/keras28_5_save_model_weights2.h5")

# 14.127509117126465
# loss : 14.127509117126465
# r2 : 0.8381270211450809

end_time = time.time()

# predict
loss = model.evaluate(x_test, y_test, verbose = 0)

print(loss)

y_predict = model.predict(x_test)

# print(y_predict)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("r2 :", r2)
print("fit time :", round(end_time - start_time, 2), "초")