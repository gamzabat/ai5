import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
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

#2 model
model = Sequential()

# model.add(Conv2D(32, 3, input_shape = (100, 100, 3), activation = 'relu', padding = 'same'))
# model.add(Conv2D(32, 3, activation = 'relu', padding = 'same'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.3))
# model.add(Conv2D(32, 2, activation = 'relu', padding = 'same'))
# model.add(MaxPooling2D())
# model.add(Conv2D(16, 2, activation = 'relu', padding = 'same'))
# model.add(Flatten())
# model.add(Dense(16, activation = 'relu'))
# model.add(Dense(16, activation = 'relu'))
# model.add(Dropout(0.3))
# model.add(Dense(16, activation = 'relu'))

# model.add(Dense(1, activation = 'sigmoid'))

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3), padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(filters=128, activation='relu', kernel_size=(3,3), padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu', padding='same')) 
model.add(MaxPooling2D())
model.add(Dropout(0.25))

model.add(Flatten()) 
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu')) 
model.add(Dropout(0.25))
model.add(Dense(512, activation='relu')) 
model.add(Dropout(0.25))

model.add(Dense(1, activation='sigmoid'))

#3 compile
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

##################### mcp 세이브 파일명 만들기 시작 ###################
import datetime

date = datetime.datetime.now()

date = date.strftime("%y%m%d_%H%M%S") # 240726_165505

PATH = './_save/keras42/kaggle_cat_dog/'

filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

filepath = ''.join([PATH, 'k42_', date, "_", filename])
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
    patience = 30,
    restore_best_weights = True
)

start_time = time.time()

model.fit(
    x_train,
    y_train,
    validation_split = 0.1,
    callbacks = [es, mcp],
    epochs = 1024,
    batch_size = 16,
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

sample_submission_csv.to_csv(PATH_SUBMISSION + "sampleSubmission_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".csv")

# loss : [0.33697962760925293, 0.8543000221252441] -> 0.30366
# acc : 0.8543
# fit time 721.47 sec

# loss : [0.46252983808517456, 0.7623000144958496] -> 0.29466
# acc : 0.7623