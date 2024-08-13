import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU, Bidirectional

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

#2 model
model = Sequential()

# model.add(Bidirectional(SimpleRNN(units = 10), input_shape = (3, 1))) # 120
# model.add(SimpleRNN(units = 10, input_shape = (3, 1))) # 240

# model.add(Bidirectional(LSTM(units = 10), input_shape = (3, 1))) # 960
# model.add(LSTM(units = 10, input_shape = (3, 1))) # 480

# model.add(Bidirectional(GRU(units = 10), input_shape = (3, 1))) # 780
# model.add(GRU(units = 10, input_shape = (3, 1))) # 390 

model.add(Dense(7))
model.add(Dense(1))

model.summary()