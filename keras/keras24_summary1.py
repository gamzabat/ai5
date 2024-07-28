from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#2. model
# https://mizzlena.tistory.com/32

model = Sequential()

model.add(Dense(3, input_dim = 1))
model.add(Dense(5))
model.add(Dense(5))

model.add(Dense(1))

model.summary()