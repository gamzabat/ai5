import numpy as np

a = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]).T

print(a)

size = 6

def split_x(dataset, size):
    result = []

    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]

        result.append(subset)

    return np.array(result)

x_split = split_x(a, size)

print('---------------------------')
print(x_split)
print('---------------------------')
print(x_split.shape)

x = x_split[:, :-1]
y = x_split[:, -1, 0]

print('---------------------------')
print(x)
print('---------------------------')
print(y)
print(x.shape, y.shape)