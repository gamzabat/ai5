from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.datasets import fetch_california_housing

#1 data
datasets = fetch_california_housing()

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

model.add(Dense(10, input_dim = 8))

model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))

model.add(Dense(1))

# compile
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train, y_train, epochs = 10000, batch_size = 500)

# predict
loss = model.evaluate(x_test, y_test)

print(loss)

y_predict = model.predict(x_test)

print(y_predict)

r2 = r2_score(y_test, y_predict)

print(r2) # 0.6164500002614047 : 0.7, 7755, 10000, 500