# x가 하나인데 y가 3개인 경우를 예측 -> x의 데이터가 부족하기에 성능을 보장할 수 없다

import numpy as np

from keras.models import Sequential
from keras.layers import Dense

#1 data
x = np.array([range(10)]).T # x1, x2, x3 데이터

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            , [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
            , [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]).T # y1, y2, y3 데이터

print(x.shape)
print(y.shape)

#2 model
model = Sequential()

model.add(Dense(3, input_dim = 1)) # 1개 input을 3개로
model.add(Dense(10))
model.add(Dense(3))

#3 compile and training
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x, y, epochs = 1000, batch_size = 1)

#4 predict
loss = model.evaluate(x, y)
result = model.predict([[10]])

print("loss : ", loss)
print("result : ", result)

# loss :  3.477375543070593e-08
# result :  [[ 1.1000044e+01  3.4761429e-04 -9.9962586e-01]]