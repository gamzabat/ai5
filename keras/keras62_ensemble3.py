import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Input, Concatenate, concatenate
from sklearn.metrics import r2_score

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#1 data
x1_datesets = np.array([range(100), range(301, 401)]).T
                       # 삼성 종가  하이닉스 종가
x2_datasets = np.array([range(101, 201), range(411, 511), range(150, 250)]).T
                       # 원유, 환율, 금시세
x3_datasets = np.array([range(100), range(301, 401, ),
                        range(77, 177), range(33, 133)]).T

y1 = np.array(range(3001, 3101)) # 한강의 온도
y2 = np.array(range(13001, 13101)) # 비트코인의 가격

x1_train, x1_test, x2_train, x2_test, x3_train, x3_test,\
    y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1_datesets,
    x2_datasets,
    x3_datasets,
    y1,
    y2,
    train_size = 0.9,
    random_state = 7777
)

#2-1 model
input1 = Input(shape=(2, ))

dense1 = Dense(10, activation = 'relu', name = 'bit1')(input1)
dense2 = Dense(10, activation = 'relu', name = 'bit2')(dense1)
dense3 = Dense(10, activation = 'relu', name = 'bit3')(dense2)
dense4 = Dense(10, activation = 'relu', name = 'bit4')(dense3)
output1 = Dense(10, activation = 'relu', name = 'bit5')(dense4)

#2-2 model
input11 = Input(shape=(3, ))

dense11 = Dense(10, activation = 'relu', name = 'bit11')(input11)
dense21 = Dense(10, activation = 'relu', name = 'bit21')(dense11)
output11 = Dense(10, activation = 'relu', name = 'bit10')(dense21)

#2-3 model
input31 = Input(shape=(4, ))

dense31 = Dense(10, activation = 'relu', name = 'bit31')(input31)
dense32 = Dense(10, activation = 'relu', name = 'bit32')(dense31)
dense33 = Dense(10, activation = 'relu', name = 'bit33')(dense32)
output31 = Dense(10, activation = 'relu', name = 'bit35')(dense33)

merge1 = Concatenate(name = 'merge1')([output1, output11, output31])

last_output = Dense(10, name = 'last')(merge1)

divide1 = Dense(10)(last_output)
divide1 = Dense(10)(divide1)
divide1 = Dense(1)(divide1)

divide2 = Dense(10)(last_output)
divide2 = Dense(10)(divide2)
divide2 = Dense(1)(divide2)

model = Model(inputs = [input1, input11, input31], outputs = [divide1, divide2])

model.summary()

#3 compile, fit
model.compile(loss = 'mse', optimizer = 'adam')

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()
date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras62/ensemble3/'

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
    patience = 100,
    restore_best_weights = True
)

hist = model.fit(
    [x1_train, x2_train, x3_train],
    [y1_train, y2_train],
    validation_split = 0.1,
    callbacks = [es, mcp],
    batch_size = 3,
    epochs = 1500
)

#4 predict
x1_pre = np.array([range(3101, 3106), range(101, 106)]).T
                       # 삼성 종가  하이닉스 종가
x2_pre = np.array([range(3301, 3306), range(2501, 2506), range(4001, 4006)]).T
                       # 원유, 환율, 금시세
x3_pre = np.array([range(5001, 5006), range(6001, 6006), range(5501, 5506), range(6501, 6506)]).T
                       # 원유, 환율, 금시세

loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test])

print("loss :", loss)

result = model.predict([x1_pre, x2_pre, x3_pre])

print("예측값 :", result)

# loss : [0.0012566030491143465, 2.3502110707340762e-05, 0.0012331008911132812]