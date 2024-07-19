import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]).T

y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(x.shape) # (10, 3)
print(y.shape) # (10,)

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
epochs = 500

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = epochs, batch_size = 3)

#4 평가, 예측
loss = model.evaluate(x, y)

print("==========================================================================")
print("epochs : ", epochs)
print("로스 : ", loss)

result = model.predict([[10, 1.3, 0]])

print("[10, 1.3, 0]의 예측값\n", result)