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
    train_size = 0.80,
    random_state = 7777
)

# [실습] : R2 0.52 이상

#2 model
model = Sequential()

model.add(Dense(100, input_dim = 10, activation = 'relu'))

model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))

model.add(Dense(1))

#3 compile
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train,
          y_train,
          validation_split = 0.2,
          epochs = 300,
          batch_size = 512,
          verbose = 1)

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("r2 :", r2)

# validation_split : 0.2
# loss : 2760.5927734375
# r2 : 0.5708649436766161

# validation_split : none
# 0.6091238532763392 : 0.70 7777 30000 50