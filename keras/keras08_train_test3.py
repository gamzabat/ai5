import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.7, # default : 0.75
    shuffle = True, # default : True
    random_state = 7979
)

print('x_train :', x_train)
print('x_test :', x_test)
print('y_train :', y_train)
print('y_test :', y_test)

# x_train : [ 6  9  4  2  7 10  3]
# x_test : [5 1 8]
# y_train : [ 6  9  4  2  7 10  3]
# y_test : [5 1 8]
[]
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