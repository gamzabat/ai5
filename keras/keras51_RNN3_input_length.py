import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU

#1 data
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x = np.array(
   [[1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9]])

y = np.array([4, 5, 6, 7, 8, 9, 10])

x = x.reshape(x.shape[0], x.shape[1], 1)

print(x.shape, y.shape)
print(x)

# 3-D tensor with shape(batch_size, timesteps, features)

#2 model
model = Sequential()

#model.add(SimpleRNN(units = 10, input_shape = (3, 1), activation = 'tanh')) # 3 : timesteps, 1 : features
model.add(SimpleRNN(units = 10, input_length = 3, input_dim = 1, activation = 'tanh')) # 3 : timesteps, 1 : features

model.add(Dense(7))
model.add(Dense(1))

#3 compile, fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x, y, epochs = 10)

results = model.evaluate(x, y)

print("loss :", results)

y_pred = model.predict(np.array([8, 9, 10]).reshape(1, 3, 1)) # [[[8], [9], [10]]]

print("pred :", y_pred)