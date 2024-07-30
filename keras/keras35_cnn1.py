from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

model = Sequential()

model.add(Conv2D(10, (2, 2), input_shape = (5, 5, 1))) # 가로 세로 컬럼(1은 컬러수) shape : (4, 4, 10)
model.add(Conv2D(5, (2, 2))) # shape : (3, 3, 5)

model.summary()