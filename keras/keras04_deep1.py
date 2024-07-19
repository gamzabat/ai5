from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1 데이터
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])

#[실습] 레이어의 깊이와 노드의 갯수를 이용해서 [6]을 만든다
# epoch는 100으로 고정
# 소수 넷째자리까지 맞추면 합격. 예: 6.00 or 5.9999

#2 모델 구성
model = Sequential()

model.add(Dense(1, input_dim = 1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(60))
model.add(Dense(70))
model.add(Dense(80))
model.add(Dense(90))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

epochs = 100

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = epochs)

#4 평가, 예측
loss = model.evaluate(x, y)

print("==========================================================================")
print("epochs : ", epochs)
print("로스 : ", loss)

result = model.predict([6])

print("6의 예측값 : ", result)