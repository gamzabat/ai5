from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train / 255.
x_test = x_test / 255.

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
print(x_train[0].shape) # (32, 32)

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
    save_to_dir="./_data/_save_img/04_cifar100/"
).next()[0]