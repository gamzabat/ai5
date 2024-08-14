import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Concatenate, concatenate
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1 data
x1_datesets = np.array([range(100), range(301, 401)]).T
                       # 삼성 종가  하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).T
                       # 원유, 환율, 금시세

y = np.array(range(3001, 3101)) # 예나 강의 온도

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datesets,
    x2_datasets,
    y,
    train_size = 0.9,
    random_state = 7777
)

#2-1 model
input1 = Input(shape=(2, ))

dense1 = Dense(10, activation = 'relu', name = 'bit1')(input1)
output1 = Dense(50, activation = 'relu', name = 'bit5')(dense1)

#2-2 model
input11 = Input(shape=(3, ))

dense11 = Dense(100, activation = 'relu', name = 'bit11')(input11)
output11 = Dense(300, activation = 'relu', name = 'bit10')(dense11)

merge1 = Concatenate(name = 'merge1')([output1, output11])

merge2 = Dense(20, name = 'merge3')(merge1)

last_output = Dense(1, name = 'last')(merge2)

model = Model(inputs = [input1, input11], outputs = last_output)

model.summary()

#3 compile, fit
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras62/ensemble1/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath = ''.join([PATH, 'k62_', date, "_", filename])
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
    patience = 16,
    restore_best_weights = True
)

hist = model.fit(
    [x1_train, x2_train],
    y_train,
    validation_split = 0.2,
    callbacks = [es, mcp],
    batch_size = 1,
    epochs = 400
)

#4 predict
# model = load_model("./_save/k62_01.hdf5")

x1_pre = np.array([range(3101, 3106), range(101, 106)]).T
                       # 삼성 종가  하이닉스 종가
x2_pre = np.array([range(3301, 3306), range(2501, 2506), range(4001, 4006)]).T
                       # 원유, 환율, 금시세

loss = model.evaluate([x1_test, x2_test], y_test)

print("loss :", loss)

result = model.predict([x1_pre, x2_pre])

print("예측값 :", result)

# loss : [0.00012958646402694285, 0.0]