# 18 - 1 copy
# 16_val1 copy

import sklearn as sk

print(sk.__version__) # 0.24.2

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.datasets import load_boston

from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

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
    random_state = 6666
)

# model
model = Sequential()

model.add(Dense(10, input_dim = 13, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))

model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))

model.add(Dense(1))

# compile
model.compile(loss = 'mse', optimizer = 'adam')

start_time = time.time()

es = EarlyStopping(
    monitor = "val_loss",
    mode = 'min', # 모르면 auto
    patience = 20,
    restore_best_weights = True
)

hist = model.fit(x_train,
          y_train,
          validation_split = 0.2,
          callbacks = [es],
          epochs = 1000,
          batch_size = 32,
          verbose = 1)

end_time = time.time()

# predict
loss = model.evaluate(x_test, y_test, verbose = 0)

print(loss)

y_predict = model.predict(x_test)

print(y_predict)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("r2 :", r2)
print("fit time :", round(end_time - start_time, 2), "초")

print("=========================== hist ============================")
print(hist)
print("=========================== hist.history ====================")
print(hist.history)
print("=========================== loss ============================")
print(hist.history['loss'])
print("=========================== val_loss ========================")
print(hist.history['val_loss'])

plt.rc("font", family = "Gulim")
plt.rc("axes", unicode_minus = False)
plt.figure(figsize = (9, 6)) # 그래프 사이즈
plt.plot(hist.history['loss'], c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], c = 'blue', label = 'val_loss')
plt.legend(loc = 'upper right')
plt.title("보스톤 loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.grid()
plt.show()

# validation_split : 0.2
# loss : 12.875835418701172
# r2 : 0.8624806655064476

# validation_split : none
# loss : 14.708605766296387
# r2 : 0.8429059092106708