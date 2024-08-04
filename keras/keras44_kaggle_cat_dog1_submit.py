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

PATH_SUBMIT = "./_data/kaggle/cat_dog/test"

PATH_SUBMISSION = "./_data/kaggle/cat_dog/"

sample_submission_csv = pd.read_csv(PATH_SUBMISSION + "sample_submission.csv", index_col = 0)

NP_PATH = "./_data/_save_npy/"

x_test = np.load(NP_PATH + "keras43_01_x_test.npy")

model = load_model('./_save/k42_1902.hdf5')

y_submit = model.predict(x_test, batch_size = 16)

sample_submission_csv['label'] = y_submit

#sample_submission_csv.to_csv(PATH_SUBMISSION + "sampleSubmission_" + datetime.datetime.now().strftime("%y%m%d_%H%M%S") + ".csv")
sample_submission_csv.to_csv(PATH_SUBMISSION + "teacher0805.csv")

# loss : [0.33697962760925293, 0.8543000221252441] -> 0.30366
# acc : 0.8543
# fit time 721.47 sec

# loss : [0.46252983808517456, 0.7623000144958496] -> 0.29466
# acc : 0.7623