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

# 정제된 데이터를 통한 훈련에 있어 '평가'의 문제가 있다
# loss를 평가하는 evaluation의 문제가 존재

# 데이터를 평가할 때는 loss값으로 평가를 한다
# x, y로 평가하는 것이 문제가 된다
# 훈련하는 데이터와 평가하는 데이터를 구분하지 않아 '과적합'의 문제가 발생한다

# '과적합'이란
# 테스트 데이터와 평가 데이터가 동일하기 때문에
# 더 이상의 제대로된 평가 값이 나오지 않는 문제가 발생한다
# 데이터가 많거나, 데이터가 너무 정제된 경우에 있어서
# 과적합으로 인해 답이 나오지 않는 문제가 발생한다

# accuracy = 정확도를 통해서 검증 가능

# 하지만 과적합의 문제에서는 훈련 데이터를 통해서 같은 평가를 할 경우에는
# 정확도가 높게 나오지만 실제적으로 정확도가 높다고 할 수 없다

# 그래서 데이터를 분할한다(train data - test data)