import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

WIDTH = 100
HEIGHT = 100

#1 data
# train_datagen = ImageDataGenerator(
#     rescale = 1. / 255,         # 데이터 스케일링
#     horizontal_flip = True,     # 수평 뒤집기
#     vertical_flip = True,       # 수직 뒤집기
#     width_shift_range = 0.1,    # 10% 수평 이동
#     height_shift_range = 0.1,   # 10% 수직 이동
#     rotation_range = 5,         # 정해진 각도로 회전
#     zoom_range = 1.2,           # 축소 또는 확대
#     shear_range = 0.7,          # 수평으로 찌그러트리기
#     fill_mode = 'nearest'       # 이미지가 이동한 후 남은 공간을 어떻게 채우는가
# )

test_datagen = ImageDataGenerator(
    rescale = 1. / 255,         # 데이터 스케일링
)

PATH_TRAIN = "./_data/kaggle/cat_dog/train"
PATH_SUBMIT = "./_data/kaggle/cat_dog/test"

PATH_SUBMISSION = "./_data/kaggle/cat_dog/"

sample_submission_csv = pd.read_csv(PATH_SUBMISSION + "sample_submission.csv", index_col = 0)

start_time = time.time()

# xy_train = train_datagen.flow_from_directory(
#     PATH_TRAIN,
#     target_size = (WIDTH, HEIGHT),
#     batch_size = 75000,
#     class_mode = 'binary',
#     color_mode = 'rgb',
#     shuffle = True
# )

lead_time_train = time.time() - start_time

print(lead_time_train)

start_time = time.time()

xy_test = test_datagen.flow_from_directory(
    PATH_SUBMIT,
    target_size = (WIDTH, HEIGHT),
    batch_size = 12500,
    class_mode = 'binary',
    color_mode = 'rgb',
    shuffle = False
)

NP_PATH = "./_data/_save_npy/"

print(xy_test[0][1].shape)

# np.save(NP_PATH + 'keras43_01_x_train.npy', arr = xy_train[0][0])
# np.save(NP_PATH + 'keras43_01_y_train.npy', arr = xy_train[0][1])
np.save(NP_PATH + 'keras43_01_x_test.npy', arr = xy_test[0][0])
np.save(NP_PATH + 'keras43_01_y_test.npy', arr = xy_test[0][1])

lead_time_test = time.time() - start_time

print(lead_time_test)