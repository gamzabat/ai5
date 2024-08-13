import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, LSTM, Conv1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time
import datetime

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

NP_PATH = "./_data/_save_npy/"

PATH_TRAIN = "./_data/kaggle/cat_dog/train"
PATH_SUBMIT = "./_data/kaggle/cat_dog/test"

PATH_SUBMISSION = "./_data/kaggle/cat_dog/"

sample_submission_csv = pd.read_csv(PATH_SUBMISSION + "sample_submission.csv", index_col = 0)

start_time = time.time()

x = np.load(NP_PATH + "keras42_good_x_train.npy")
y = np.load(NP_PATH + "keras42_good_y_train.npy")
xy_test = np.load(NP_PATH + "keras42_good_x_test.npy")

start_time = time.time()

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.9,
    stratify = y,
    random_state = 11
)

lead_time_split = time.time() - start_time

print(lead_time_split)

x_train = x_train.reshape(-1, 300, 100)
x_test = x_test.reshape(-1, 300, 100)

#2 model
model = Sequential()

model.add(Conv1D(filters = 10, kernel_size = 3, input_shape = (300, 100), activation='relu'))
model.add(Flatten()) 
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#3 compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras60/kaggle_cat_dog/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k60_', date, "_", filename])
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
    patience = 5,
    restore_best_weights = True
)

start_time = time.time()

model.fit(
    x_train,
    y_train,
    validation_split = 0.1,
    callbacks = [es, mcp],
    epochs = 1024,
    batch_size = 1024,
    verbose = 1
)

end_time = time.time()

#4 predict
loss = model.evaluate(x_test, y_test, verbose = 0, batch_size = 16)

print("end of evaluation")

y_pred = model.predict(x_test, batch_size = 16)

print("lead split time :", lead_time_split)
print("loss :", loss)
print("acc :", accuracy_score(y_test, np.round(y_pred)))
print("fit time", round(end_time - start_time, 2), "sec")

y_submit = model.predict(xy_test, batch_size = 16)
y_submit = np.clip(y_submit, 1e-6, 1-(1e-6))

sample_submission_csv['label'] = y_submit

sample_submission_csv.to_csv(PATH_SUBMISSION + "sample_submission_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".csv")

# loss : [0.46252983808517456, 0.7623000144958496] -> 0.29466 - CNN
# acc : 0.7623

# loss : [0.6631460189819336, 0.6192499995231628] - LSTM
# acc : 0.61925
# fit time 100.02 sec

# loss : [0.6358771920204163, 0.6439999938011169] - Conv1D
# acc : 0.644
# fit time 26.02 sec