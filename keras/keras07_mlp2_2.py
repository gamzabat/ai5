import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 데이터
x = np.array(range(10)) # 벡터를 반환한다

print(x) # [0 1 2 3 4 5 6 7 8 9]
print(x.shape) # (10,)

x = np.array(range(1, 11)) # 벡터를 반환한다

print(x) # [ 1  2  3  4  5  6  7  8  9 10]
print(x.shape) # (10,)

x = np.array([range(10), range(21, 31), range(201, 211)]).T

print(x)
print(x.shape) # (10, 3)

y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# [실습]
# [10, 31, 211]

#2 모델 구성
model = Sequential()

model.add(Dense(10, input_dim = 3)) # 3개의 feature
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3 컴파일, 훈련
epochs = 400

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = epochs, batch_size = 2)

#4 평가, 예측
loss = model.evaluate(x, y)

print("==========================================================================")
print("epochs : ", epochs)
print("로스 : ", loss)

result = model.predict([[10, 31, 211]])

print("[10, 31, 211]의 예측값\n", result)