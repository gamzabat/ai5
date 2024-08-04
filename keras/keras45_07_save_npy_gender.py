import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time

import os

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

WIDTH = 100
HEIGHT = 100

PATH_TRAIN = "./_data/kaggle/biggest-gender/faces"

#1 data
train_datagen1 = ImageDataGenerator(
    rescale = 1. / 255,         # 데이터 스케일링
    # horizontal_flip = True,     # 수평 뒤집기
    # vertical_flip = True,       # 수직 뒤집기
    # width_shift_range = 0.1,    # 10% 수평 이동
    # height_shift_range = 0.1,   # 10% 수직 이동
    # rotation_range = 5,         # 정해진 각도로 회전
    # zoom_range = 1.2,           # 축소 또는 확대
    # shear_range = 0.7,          # 수평으로 찌그러트리기
    # fill_mode = 'nearest'       # 이미지가 이동한 후 남은 공간을 어떻게 채우는가
)

# train_datagen2 = ImageDataGenerator(
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

start_time = time.time()

xy_train1 = train_datagen1.flow_from_directory(
    PATH_TRAIN,
    target_size = (WIDTH, HEIGHT),
    batch_size = 33000,
    class_mode = 'binary',
    color_mode = 'rgb',
    shuffle = True
)

# xy_train2 = train_datagen2.flow_from_directory(
#     PATH_TRAIN,
#     target_size = (WIDTH, HEIGHT),
#     batch_size = 33000,
#     class_mode = 'binary',
#     color_mode = 'rgb',
#     shuffle = True
# )

lead_time_train = time.time() - start_time

print(lead_time_train)

# x1 = xy_train1[0][0].reshape(-1, WIDTH * HEIGHT * 3)
# x2 = xy_train2[0][0].reshape(-1, WIDTH * HEIGHT * 3)

# x = np.concatenate((x1, x2), axis = 0).reshape(-1, WIDTH, HEIGHT, 3)
# y = np.concatenate((xy_train1[0][1], xy_train2[0][1]))

NP_PATH = './_data/_save_npy/'

np.save(NP_PATH + 'keras45_07_x_train.npy', arr = xy_train1[0][0])
np.save(NP_PATH + 'keras45_08_y_train.npy', arr = xy_train1[0][1])

lead_time_test = time.time() - start_time

print(lead_time_test)