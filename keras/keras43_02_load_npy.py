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

WIDTH = 80
HEIGHT = 80

#1 data
NP_PATH = "./_data/_save_npy/"

PATH_SUBMISSION = "./_data/kaggle/cat_dog/"

sample_submission_csv = pd.read_csv(PATH_SUBMISSION + "sample_submission.csv", index_col = 0)

start_time = time.time()

# np.save(NP_PATH + 'keras43_01_x_train.npy', arr = xy_train[0][0])
# np.save(NP_PATH + 'keras43_01_y_train.npy', arr = xy_train[0][1])
# np.save(NP_PATH + 'keras43_01_x_test.npy', arr = xy_test.next())
# np.save(NP_PATH + 'keras43_01_y_test.npy', arr = xy_test.ext()[1])

x_train = np.load(NP_PATH + "keras43_01_x_train.npy")
y_train = np.load(NP_PATH + "keras43_01_y_train.npy")
x_test = np.load(NP_PATH + "keras43_01_x_test.npy")
y_test = np.load(NP_PATH + "keras43_01_y_test.npy")

# print(x_train)
# print(x_train.shape) # (75000, 100, 100, 3)

# print(y_test)
# print(y_train.shape) # (75000,)

# print(x_test)
# print(x_test.shape) # (12500, 80, 80, 3)

print(y_test)
print(y_test.shape) # (75000, 100, 100, 3)

lead_time_test = time.time() - start_time

print(lead_time_test)