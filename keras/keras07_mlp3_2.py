import numpy as np

from keras.models import Sequential
from keras.layers import Dense

#1 data
x = np.array([range(10), range(21, 31), range (201, 211)]).T

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            , [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
            , [9, 8, 7, 6, 5, 4, 3, 2, 1 ,0]]).T # y1, y2, y3의 데이터가 존재

print(x.shape)
print(y.shape)

#2 model
model = Sequential()

model.add(Dense(10, input_dim = 3)) # input으로 3개의 데이터 묶음을 입력
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(3)) # output으로 3개의 데이터 출력

#3 compile
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x, y, epochs = 2000, batch_size = 1)

#4 predict
loss = model.evaluate(x, y)

result = model.predict([[10, 31, 211]])

print("loss : ", loss)
print("result : ", result)

# loss :  2.4894605474279352e-11
# result :  [[ 1.0999998e+01 -1.1427328e-06 -9.9999845e-01]]