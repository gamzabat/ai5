# 15_4 복사

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import time

#1 data
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.65,
    shuffle = True,
    random_state = 133
)

print(x_train, x_test)
print(y_train, y_test)

#2 model
model = Sequential()

model.add(Dense(32, input_dim = 1))

model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))

#3 compile
model.compile(loss = 'mse', optimizer = 'adam')

start_time = time.time()

model.fit(x_train,
          y_train,
          validation_split = 0.3,
          epochs = 100,
          batch_size = 1,
          verbose = 1)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

result = model.predict([18])

print("로스 :", loss)
print("18의 예측값 :", result)
print("fit time :", round(end_time - start_time, 2), "초")