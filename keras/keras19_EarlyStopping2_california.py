# 11 - 2 copy

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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

# model
model = Sequential()

model.add(Dense(10, input_dim = 8))

model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))

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

print(loss)
print(r2) # 0.6164500002614047 : 0.7, 7755, 10000, 500
print("fit time", round(end_time - start_time, 2), "초")

print("========== hist ==========")
print(hist)
print("========== hist.history ==")
print(hist.history)
print("========== loss ==========")
print(hist.history["loss"])
print("========== val_loss ======")
print(hist.history["val_loss"])

plt.rc("font", family = "Gulim")
plt.rc("axes", unicode_minus = False)
plt.figure(figsize = (9, 6))
plt.plot(hist.history["loss"], color = "red", label = "loss")
plt.plot(hist.history["val_loss"], color = "blue", label = "val_loss")
plt.legend(loc = "upper right")
plt.title("캘리포니아 loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.grid()
plt.show()