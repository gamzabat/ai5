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

model.add(SimpleRNN(units = 10, input_shape=(3, 1), activation = 'tanh')) # 3 : timesteps, 1 : features
model.add(Dense(units = 7, activation = 'relu'))
model.add(Dense(1))

model.summary()

# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 10)                120

#  dense (Dense)               (None, 7)                 77

#  dense_1 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 205
# Trainable params: 205
# Non-trainable params: 0
# _________________________________________________________________

# (파라미터 갯수) = units * (units + features + 1) = 10 * (10 + 1 + 1) = 120