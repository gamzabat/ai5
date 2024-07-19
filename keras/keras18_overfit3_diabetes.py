# 11 - 2 copy

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.datasets import load_diabetes

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

hist = model.fit(x_train,
                 y_train,
                 validation_split = 0.2,
                 epochs = 200,
                 batch_size = 32)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print(loss)
print(r2) # 0.6091238532763392 : 0.70 7777 30000 50
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
plt.title("당뇨 loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.grid()
plt.show()