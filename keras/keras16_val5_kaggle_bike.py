# https://www.kaggle.com/competitions/bike-sharing-demand

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1 data
PATH = "C:/ai5/_data/bike-sharing-demand/" # 절대경로

train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)
test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)
sampleSubmission_csv = pd.read_csv(PATH + "sampleSubmission.csv", index_col = 0)

train_dt = pd.DatetimeIndex(train_csv.index)

train_csv['day'] = train_dt.day
train_csv['month'] = train_dt.month
train_csv['year'] = train_dt.year
train_csv['hour'] = train_dt.hour
train_csv['dow'] = train_dt.dayofweek

test_dt = pd.DatetimeIndex(test_csv.index)

test_csv['day'] = test_dt.day
test_csv['month'] = test_dt.month
test_csv['year'] = test_dt.year
test_csv['hour'] = test_dt.hour
test_csv['dow'] = test_dt.dayofweek

print(train_csv.shape, test_csv.shape, sampleSubmission_csv.shape) # (10886, 11) (6493, 8) (6493, 1)
print(train_csv.columns) # ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']

train_csv.info()
test_csv.info()

print(train_csv.describe().T) # 50% : 전체의 가운데 값

########################## 결측치 확인 ##########################
print(train_csv.isna().sum())

print(test_csv.isna().sum())
#---------------------------------------------------------------
x = train_csv.drop(['casual', 'registered', 'count'], axis = 1) # [] -> python에서 list 형식이다

y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.800,
    random_state = 7777 
)

#2 model
model = Sequential()

model.add(Dense(100, input_dim = 13, activation = 'relu'))

model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(20, activation = 'relu'))

model.add(Dense(1, activation = 'linear'))

#3 compile
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x_train,
          y_train,
          validation_split = 0.4,
          epochs = 200,
          batch_size = 32,
          verbose = 1)

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

y_submit = model.predict(test_csv)

print(y_submit)
print(y_submit.shape)

sampleSubmission_csv['count'] = y_submit

print(sampleSubmission_csv)
print(sampleSubmission_csv.shape)

print("loss :", loss)
print("r2 :", r2)

# validation_split : 0.2
# loss : 5640.92724609375
# r2 : 0.8288752397788648

# validation_split : none
# loss : 20513.6875 0.80 7777 300 128

sampleSubmission_csv.to_csv(PATH + "sampleSubmission_0718.csv")