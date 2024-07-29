from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes

from keras.callbacks import EarlyStopping, ModelCheckpoint

#1 data
datasets = load_diabetes()

x = datasets.data
y = datasets.target

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
# model = Sequential()

# model.add(Dense(32, input_shape = (10,), activation = 'relu'))

# model.add(Dropout(0.3))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(32, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(16, activation = 'relu'))

# model.add(Dense(1))

input1 = Input(shape = (10,))

dense1 = Dense(32, activation = 'relu')(input1)
drop1 = Dropout(0.3)(dense1)
dense2 = Dense(32, activation = 'relu')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(32, activation = 'relu')(drop2)
drop3 = Dropout(0.3)(dense3)
dense4 = Dense(16, activation = 'relu')(drop3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(16, activation = 'relu')(drop4)
drop5 = Dropout(0.3)(dense5)
dense6 = Dense(16, activation = 'relu')(drop5)
drop6 = Dropout(0.3)(dense6)
dense7 = Dense(16, activation = 'relu')(drop6)
output1 = Dense(1)(dense7)

model = Model(inputs = input1, outputs = output1)

#3 compile
model.compile(loss = 'mse', optimizer = 'adam')

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

print(date) # 2024-07-26 16:49:36.336699
print(type(date)) # <class 'datetime.datetime'>

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras32/03-diabetes/'

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
    monitor = "val_loss",
    mode = "min",
    patience = 64,
    restore_best_weights = True
)

hist = model.fit(x_train,
                 y_train,
                 validation_split = 0.2,
                 callbacks = [es, mcp],
                 epochs = 1000,
                 batch_size = 32)

#4 predict
loss = model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)

print("loss :", loss)
print("rs :", r2)

# loss : 2436.35791015625
# rs : 0.5897085199507599