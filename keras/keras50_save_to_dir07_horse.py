from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import numpy as np

NP_PATH = "./_data/_save_npy/"

x = np.load(NP_PATH + "keras44_02_x_train.npy")
y = np.load(NP_PATH + "keras44_02_y_train.npy")

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size = 0.8,
    shuffle = False
)

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

print(x_train.shape[0])

randidx = np.random.randint(x_train.shape[0], size = augment_size)

print(np.min(randidx), np.max(randidx))
print(x_train[0].shape) # (100, 100)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

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
    save_to_dir="./_data/_save_img/07_horse/"
).next()[0]