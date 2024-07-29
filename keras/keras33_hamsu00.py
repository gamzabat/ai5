import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout

#1 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]).T

y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

#2-1 모델구성 (순차형)
model = Sequential()

model.add(Dense(10, input_shape = (3,)))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Dense(1))

model.summary()

# _________________________________________________________________
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  dense (Dense)               (None, 10)                40

#  dense_1 (Dense)             (None, 9)                 99

#  dense_2 (Dense)             (None, 8)                 80

#  dense_3 (Dense)             (None, 7)                 63

#  dense_4 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 290
# Trainable params: 290
# Non-trainable params: 0
# _________________________________________________________________

#2-2 모델구성 (함수형)
input1 = Input(shape = (3,))

dense1 = Dense(10, name = 'my_layer_1')(input1)
dense2 = Dense(9, name = 'my_layer_2')(dense1)
drop1 = Dropout(0.3)(dense2)
dense3 = Dense(8, name = 'my_layer_3')(drop1)
drop2 = Dropout(0.2)(dense3)
dense4 = Dense(7, name = 'my_layer_4')(drop2)
output1 = Dense(1)(dense4)

model = Model(inputs = input1, outputs = output1)

model.summary()

# _________________________________________________________________
# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 3)]               0

#  dense_1 (Dense)             (None, 10)                40

#  dense_2 (Dense)             (None, 9)                 99

#  dense_3 (Dense)             (None, 8)                 80

#  dense_4 (Dense)             (None, 7)                 63

#  dense_5 (Dense)             (None, 1)                 8

# =================================================================
# Total params: 290
# Trainable params: 290
# Non-trainable params: 0
# _________________________________________________________________