import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1 데이터
x = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]).T # 2개의 feature
#x = np.array([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]]) # 2개의 feature

#x = x.transpose()
#x = np.transpose(x)
#x = x.reshape(5, 2)

y = np.array([1, 2, 3, 4, 5])

print(x.shape) # (5, 2)
print(y.shape) # (5,)

#2 모델 구성
model = Sequential()

model.add(Dense(10, input_dim = 2)) # 2개의 feature
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1)) # y의 형태가 벡터 1개이므로

#3 컴파일, 훈련
epochs = 100

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = epochs, batch_size = 2)

#4 평가, 예측
loss = model.evaluate(x, y)

print("==========================================================================")
print("epochs : ", epochs)
print("로스 : ", loss)

result = model.predict([[6, 11], [7, 12], [102, 107]])

print("[6, 11]의 예측값\n", result)

# [실습] : 소수 2째자리까지 맞춘다