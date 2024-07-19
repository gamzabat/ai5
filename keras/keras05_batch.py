from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1 데이터
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 2, 3, 5, 4, 6])

#[실습]
# 최소의 loss를 만드시오
# batch_size 조절
# 애포 100은 자유
# 로스 기준 0.32 미만

#2 모델 구성
model = Sequential()

model.add(Dense(1, input_dim = 1))
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
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

epochs = 10

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = epochs, batch_size = 2)

#4 평가, 예측
loss = model.evaluate(x, y)

print("==========================================================================")
print("epochs : ", epochs)
print("로스 : ", loss)

result = model.predict([7])

print("7의 예측값 : ", result)