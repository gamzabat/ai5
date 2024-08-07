import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

#1 data
PATH_TRAIN = "./_data/image/cat_and_dog/train/"
PATH_SUBMIT = "./_data/image/cat_and_dog/test/"

train_image_datagen = ImageDataGenerator(
    rescale = 1. / 255,           # 데이터 스케일링
)

xy_image_train = train_image_datagen.flow_from_directory(
    PATH_TRAIN,
    target_size = (100, 100),
    batch_size = 20000,
    class_mode = 'binary',
    color_mode = 'rgb',
    shuffle = True
)

print(xy_image_train[0][0].shape, xy_image_train[0][1].shape)

NP_PATH = "./_data/_save_npy/"

PATH_TRAIN = "./_data/kaggle/cat_dog/train"
PATH_SUBMIT = "./_data/kaggle/cat_dog/test"

PATH_SUBMISSION = "./_data/kaggle/cat_dog/"

sample_submission_csv = pd.read_csv(PATH_SUBMISSION + "sample_submission.csv", index_col = 0)

x = np.load(NP_PATH + "keras42_good_x_train.npy")
y = np.load(NP_PATH + "keras42_good_y_train.npy")
xy_test = np.load(NP_PATH + "keras42_good_x_test.npy")

print(type(x))

print(x.shape, y.shape)

x = np.concatenate((x, xy_image_train[0][0]))
y = np.concatenate((y, xy_image_train[0][1]))

train_datagen = ImageDataGenerator(
    rescale = 1.0 / 255,
    horizontal_flip = True,        # 수평 뒤집기 = 증폭(완전 다른 데이터가 하나 더 생겼다)
    vertical_flip = True,         # 수직 뒤집기 = 증폭
    # width_shift_range = 0.1,      # 평행 이동   = 증폭
    # height_shift_range = 0.1,     # 평행 이동 수직
    rotation_range = 15,           # 정해진 각도만큼 이미지 회전
    # zoom_range = 1.2,             # 축소 또는 확대
    # shear_range = 0.7,            # 좌표 하나를 고정시키고 다른 몇개의 좌표를 이동시키는 변환.
    fill_mode = 'nearest',        # 원래 있던 가까운 놈으로 채운다.
)

augment_size = 5000

randidx = np.random.randint(x.shape[0], size = augment_size)

print(np.min(randidx), np.max(randidx))
print(x[0].shape) # (100, 100)

x_augmented = x[randidx].copy()
y_augmented = y[randidx].copy()

print(x_augmented.shape, y_augmented.shape)

x_augmented = x_augmented.reshape(
    x_augmented.shape[0],
    x_augmented.shape[1],
    x_augmented.shape[2], 3)

print(x_augmented.shape)

x_augmented = train_datagen.flow(
    x_augmented,
    y_augmented,
    batch_size = augment_size,
    shuffle = False,
    save_to_dir="./_data/_save_img/05_cat_dog/"
).next()[0]