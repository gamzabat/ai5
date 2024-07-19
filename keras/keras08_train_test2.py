import numpy as np

from keras.models import Sequential
from keras.layers import Dense

# data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8 ,9, 10])
y = np.array([1, 2, 3, 4, 5, 6 ,7, 8, 9, 10])

#[실습] 넘파이 리스트를 7:3으로 슬라이싱

x_train = x[:7].T # x[0:7], x[:-3]
y_train = y[:7].T

x_test = x[7:].T
y_test = y[7:].T

# model
model = Sequential()

model.add(Dense(1, input_dim = 1))

# compile
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train, y_train, epochs = 1000, batch_size = 1)

# predict
loss = model.evaluate(x_test, y_test)
result = model.predict([11])

print(loss)
print(result)

# 7.579122740649855e-14
# [[11.]]