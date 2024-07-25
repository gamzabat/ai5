# 11 - 2 copy

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

from sklearn.datasets import load_diabetes

from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import time

#1 data
datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x)
print(y)

print(x.shape, y.shape) # (442, 10), (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.70,
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

# [실습] : R2 0.52 이상

#2 model
model = Sequential()

model.add(Dense(10, input_dim = 10))

model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))

model.add(Dense(1))

#3 compile
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

#4 predict
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("rs :", r2)
print("fit time", round(end_time - start_time, 2), "초")

# before scaler
# loss : 2446.24072265625
# rs : 0.5880442026818675

# after minMaxScaler
# loss : 2414.8505859375
# rs : 0.5933304672650037

# after standardScaler
# loss : 2508.95068359375
# rs : 0.5774836342690469

# after minAbsScaler
# loss : 2513.2109375
# rs : 0.5767661073413191

# after robustScaler
# loss : 2540.544189453125
# rs : 0.5721631541589456