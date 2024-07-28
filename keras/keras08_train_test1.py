import numpy as np

from keras.models import Sequential
from keras.layers import Dense

# data
x_train = np.array([1, 2, 3, 4, 5, 6, 7])
y_train = np.array([1, 2, 3, 4, 5, 6, 7])

x_test = np.array([8, 9, 10])
y_test = np.array([8, 9, 10])

# model
model = Sequential()

model.add(Dense(10, input_dim = 1))
model.add(Dense(20))
model.add(Dense(1))

# compile
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train, y_train, epochs = 1000, batch_size =3)

# predict
print("=============================================================")

loss = model.evaluate(x_test, y_test)
result = model.predict([11])

print("loss : ", loss)
print("result : ", result)

# loss :  2.1979456015647214e-12
# result :  [[11.]]