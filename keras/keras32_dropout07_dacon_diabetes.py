# https://dacon.io/competitions/official/236068/overview/description

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PATH = "./_data/dacon/diabetes/" # 상대경로

#1 data
train_csv = pd.read_csv(PATH + "train.csv", index_col = 0)

print(train_csv) # [1459 rows x 10 columns]

test_csv = pd.read_csv(PATH + "test.csv", index_col = 0)

print(test_csv) # [715 rows x 9 columns]

submission_csv = pd.read_csv(PATH + "sample_submission.csv", index_col = 0)

print(submission_csv) # [715 rows x 1 columns] - 결측치

print(test_csv.isna().sum())

train_csv.info() # Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome

test_csv.info()

print(train_csv.shape)

train_csv = train_csv[train_csv['BloodPressure'] != 0]
train_csv = train_csv[train_csv['BMI'] > 0.0]
train_csv = train_csv[train_csv['Glucose'] > 0]
 
print(train_csv['Glucose'].value_counts())

print(train_csv.shape)

train_csv.info()

x = train_csv.drop(['Outcome'], axis = 1)

y = train_csv['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.70,
    random_state = 7777
)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)

x_train = min_max_scaler.transform(x_train)
x_test = min_max_scaler.transform(x_test)

#2 model
model = Sequential()

model.add(Dense(16, input_dim = 8, activation = 'relu'))

model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

#3 compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras32/07-dacon-diabetes/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k32_', date, "_", filename])
##################### mcp 세이브 파일명 만들기 끝 ###################

mcp = ModelCheckpoint(
    monitor = 'val_loss',
    mode = 'auto',
    verbose = 1,
    save_best_only = True,
    filepath = filepath
)

es = EarlyStopping( 
    monitor = 'val_loss',
    mode = 'min',
    patience = 50,
    restore_best_weights = True
)

hist = model.fit(
    x_train,
    y_train,
    validation_split = 0.3,
    callbacks = [es, mcp],
    epochs = 1000,
    batch_size = 4,
    verbose = 2
)

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0)

y_pred = model.predict(x_test)

print(y_pred[:30])

print(np.round(y_pred[:30]))

print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))

y_submit = model.predict(test_csv)

submission_csv['Outcome'] = np.round(y_submit)

submission_csv.to_csv(PATH + "sample_submission_0726.csv")

# before dropout
# loss : [0.5278410911560059, 0.725806474685669]
# acc : 0.7258064516129032

# after dropout
# loss : [0.5130333304405212, 0.7243243455886841]
# acc : 0.7243243243243244