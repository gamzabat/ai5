import numpy as np

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split

x = np.array(range(1, 17))
y = np.array(range(16, 0, -1))

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.80
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    train_size = 0.75
)

x_train.sort()
x_val.sort()
x_test.sort()

y_train.sort()
y_val.sort()
y_test.sort()

print(x_train, x_val, x_test)
print(y_train, y_val, y_test)

# # model
# model = Sequential()

# model.add(Dense(10, input_dim = 1))
# model.add(Dense(20))
# model.add(Dense(1))

# # compile
# model.compile(loss = 'mse', optimizer = 'adam')

# model.fit(x_train,
#           y_train,
# #          validation_data = (x_val, y_val), # 이 부분만 추가됨
#           epochs = 600,
#           batch_size = 3,
#           verbose = 1)

# # verbose = 0 : 침묵
# # verbose = 1 : 디폴트
# # verbose = 2 : 프로그레스바 삭제
# # verbose = 나머지 : 에폭만 나옴

# # predict
# print("=============================================================")

# loss = model.evaluate(x_test, y_test)
# result = model.predict([17])

# print("loss : ", loss)
# print("result : ", result)

# # loss :  2.1979456015647214e-12
# # result :  [[11.]]