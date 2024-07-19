from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.datasets import load_diabetes

#1 data
datasets = load_diabetes()

x = datasets.data
y = datasets.target

print(x)
print(y)

print(x.shape, y.shape) # (442, 10), (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.70,
    random_state = 7777
)

# [실습] : R2 0.52 이상

#2 model
model = Sequential()

model.add(Dense(10, input_dim = 10))

model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))

model.add(Dense(1))

#3 compile
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train, y_train, epochs = 10000, batch_size = 25)

#4 predict
loss = model.evaluate(x_test, y_test)

print(loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print(r2) # 0.6091238532763392 : 0.70 7777 30000 50