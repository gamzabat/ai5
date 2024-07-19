from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.datasets import fetch_california_housing

#1 data
datasets = fetch_california_housing()

print(datasets)

x = datasets.data
y = datasets.target

print(x)
print(y)

print(x.shape, y.shape) # (20640, 8), (20640,)

# [실습] : R2 0.49 이상

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.7,
    random_state = 7755
)

# model
model = Sequential()

model.add(Dense(10, input_dim = 8, activation = 'relu'))

model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))

model.add(Dense(1))

# compile
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train,
          y_train,
          validation_split = 0.20,
          epochs = 300,
          batch_size = 256,
          verbose = 1)

# predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_predict = model.predict(x_test)

print(y_predict)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("r2 :", r2)

# validation_split : 0.2
# loss : 0.45895639061927795
# r2 : 0.6619855002100521

# validation_split : none
# r2 : 0.6164500002614047 : 0.7, 7755, 10000, 500