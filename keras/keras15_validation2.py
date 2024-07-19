import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train = x[0:10]
x_val = x[10:13]
x_test = x[13:17]

print(x_train, x_val, x_test)

y_train = y[0:10]
y_val = y[10:13]
y_test = y[13:17]

print(y_train, y_val, y_test)

# model
model = Sequential()

model.add(Dense(10, input_dim = 1))
model.add(Dense(20))
model.add(Dense(1))

# compile
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train,
          y_train,
#          validation_data = (x_val, y_val), # 이 부분만 추가됨
          epochs = 600,
          batch_size = 3,
          verbose = 1)

# verbose = 0 : 침묵
# verbose = 1 : 디폴트
# verbose = 2 : 프로그레스바 삭제
# verbose = 나머지 : 에폭만 나옴

# predict
print("=============================================================")

loss = model.evaluate(x_test, y_test)
result = model.predict([17])

print("loss : ", loss)
print("result : ", result)

# loss :  2.1979456015647214e-12
# result :  [[11.]]