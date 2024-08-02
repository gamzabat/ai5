import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1. / 255,         # 데이터 스케일링
    horizontal_flip = True,     # 수평 뒤집기
    vertical_flip = True,       # 수직 뒤집기
    width_shift_range = 0.1,    # 10% 수평 이동
    height_shift_range = 0.1,   # 10% 수직 이동
    rotation_range = 5,         # 정해진 각도로 회전
    zoom_range = 1.2,           # 축소 또는 확대
    shear_range = 0.7,          # 수평으로 찌그러트리기
    fill_mode = 'nearest'       # 이미지가 이동한 후 남은 공간을 어떻게 채우는가
)

test_datagen = ImageDataGenerator(
    rescale = 1. / 255,         # 데이터 스케일링
)

PATH_TRAIN = "./_data/image/brain/train/"
PATH_TEST = "./_data/image/brain/test/"

# 10, 200, 200, 1
xy_train = train_datagen.flow_from_directory(
    PATH_TRAIN,
    target_size = (200, 200),
    batch_size = 10,
    class_mode = 'binary',
    color_mode = 'grayscale',
    shuffle = True
) # Found 160 images belonging to 2 classes.

# 10, 200, 200, 1
xy_test = test_datagen.flow_from_directory(
    PATH_TEST,
    target_size = (200, 200),
    batch_size = 10,
    class_mode = 'binary',
    color_mode = 'grayscale'
) # Found 160 images belonging to 2 classes.

# print(xy_train[0][0].shape) # (10, 200, 200, 1)
# print(xy_train[0][1].shape) # (10,)
# print(xy_train[16]) # Asked to retrieve element 16, but the Sequence has length 16
# print(xy_train[15][3]) # IndexError: tuple index out of range

# print(type(xy_train)) # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) # <class 'tuple'>
# print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
# print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

