from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets import load_boston

# data
dataset = load_boston()

print(dataset)
print(dataset.DESCR)
print(dataset.feature_names) # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

x = dataset.data
y = dataset.target

print(x)
print(x.shape) # (506, 13)

print(y)
print(y.shape) # (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.7,
    random_state = 7777
)

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train)

x_train = min_max_scaler.transform(x_train)
x_test = min_max_scaler.transform(x_test)

# model
model = Sequential()

model.add(Dense(64, input_dim = 13, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.1))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1))

# compile
model.compile(loss = 'mse', optimizer = 'adam')

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras32/01-boston/'

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

es = EarlyStopping(monitor = 'val_loss',
                   mode = 'min',
                   patience = 10,
                   verbose = 1,
                   restore_best_weights = True)

hist = model.fit(x_train,
          y_train,
          validation_split = 0.2,
          epochs = 1000,
          callbacks = [es, mcp],
          batch_size = 32,
          verbose = 1)

model.save('./_save/keras32/keras32_save_model.h5')

# predict
loss = model.evaluate(x_test, y_test, verbose = 0)

print(loss)

y_predict = model.predict(x_test)

# print(y_predict)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("r2 :", r2)

# loss : 14.791027069091797
# r2 : 0.8305244270916647