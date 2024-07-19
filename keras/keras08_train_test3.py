import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# [검색] train과 test를 섞어서 7:3으로 나눠라
# 힌트 : 사이킷런

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.7, # default : 0.75
    shuffle = True, # default : True
    random_state = 7979)
# random_state = 123 -> 난수사전에서 123번째 난수형식으로 랜덤을 뽑는다
# random_state가 같으면 랜덤값이 고정된다
# shuffle이 True일때만 위의 내용이 적용된다
# random_state에 따라 결과가 더 좋은 경우가 있다
# (train_size + test_size)는 1을 넘을 수는 없고 1 이하이면 소실시킨다

print('x_train :', x_train)
print('x_test :', x_test)
print('y_train :', y_train)
print('y_test :', y_test)

# x_train : [ 6  9  4  2  7 10  3]
# x_test : [5 1 8]
# y_train : [ 6  9  4  2  7 10  3]
# y_test : [5 1 8]

# model
model = Sequential()

model.add(Dense(1, input_dim = 1))

# compile
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train, y_train, epochs = 1500, batch_size = 1)

# predict
loss = model.evaluate(x_test, y_test)
result = model.predict([11])

print(loss)
print(result)