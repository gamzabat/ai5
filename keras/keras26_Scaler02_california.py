# 11 - 2 copy

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from sklearn.datasets import fetch_california_housing

from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import time

#1 data
datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(x)
print(y)

print(x.shape, y.shape) # (20640, 8), (20640,)

# [실습] : R2 0.49 이상

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.7,
    random_state = 7755
)

# min_max_scaler = MinMaxScaler()

# min_max_scaler.fit(x_train)

# x_train = min_max_scaler.transform(x_train)
# x_test = min_max_scaler.transform(x_test)

# standard_scaler = StandardScaler().fit(x_train)

# x_train = standard_scaler.transform(x_train)
# x_test = standard_scaler.transform(x_test)

# max_abs_scaler = MaxAbsScaler().fit(x_train)

# x_train = max_abs_scaler.transform(x_train)
# x_test = max_abs_scaler.transform(x_test)

robust_scaler = RobustScaler().fit(x_train)

x_train = robust_scaler.transform(x_train)
x_test = robust_scaler.transform(x_test)

# model
model = Sequential()

model.add(Dense(16, input_dim = 8))

model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))

model.add(Dense(1))

# compile
model.compile(loss = 'mse', optimizer = 'adam')

start_time = time.time()

es = EarlyStopping(
    monitor = "val_loss",
    mode = "min",
    patience = 10,
    restore_best_weights = True
)

hist = model.fit(x_train,
                 y_train,
                 validation_split = 0.2,
                 callbacks = [es],
                 epochs = 1000,
                 batch_size = 32)

end_time = time.time()

# predict
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

print(y_predict)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("rs :", r2)
print("fit time", round(end_time - start_time, 2), "초")

# before scaler
# loss : 0.6575636267662048
# rs : 0.515714085272464

# after minMaxScaler
# loss : 0.5226116180419922
# rs : 0.6151043749628627

# after standardScaler
# loss : 0.5188429355621338
# rs : 0.6178800180778683

# after minAbsScaler
# loss : 0.5734553933143616
# rs : 0.5776586208497343

# after robustScaler
# loss : 1.1781790256500244
# rs : 0.13228900970696733