from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

#1 데이터
x = np.array([1, 2, 3, 4, 5, 6]) # x 데이터를 추가함
y = np.array([1, 2, 3, 4, 5, 6]) # y 데이터를 추가함

#2 모델 구성
model = Sequential()
model.add(Dense(1, input_dim = 1))

#3 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 5000)

#4 평가, 예측
loss = model.evaluate(x, y) # loss 값 출력을 추가함

print("로스 : ", loss)

result = model.predict([1, 2, 3, 4, 5, 6, 7]) # 평가/예측 값을 추가함

print("7의 예측 : ", result)