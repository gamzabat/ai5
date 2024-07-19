import numpy as np

from keras.models import Sequential
from keras.layers import Dense

#1 데이터
x = np.array([range(10), range(21, 31), range(201, 211)]).T
y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]).T #y1, #y2

print(x.shape) # (10, 3)
print(y.shape) # (10, 2)

# [실습] [10, 31, 211]

#2 모델 구성
model = Sequential()

model.add(Dense(10, input_dim = 3))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(2)) # y1, y2 값으로 출력

model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x , y, epochs = 100, batch_size = 1)

#3 예측 및 출력
loss = model.evaluate(x, y)

result = model.predict([[10, 31, 211]])

print("로스 : ", loss)
print("예측 : ", result)

#로스 :  0.0017874985933303833
#예측 :  [[11.049348   -0.06413767]]