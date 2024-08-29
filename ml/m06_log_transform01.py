import numpy as np
import matplotlib.pyplot as plt

data = np.random.exponential(scale = 2.0, size = 1000)

print(data)
print(data.shape)
print(np.min(data), np.max(data))

log_data = np.log1p(data)

plt.subplot(1, 2, 1)
plt.hist(data, bins = 50, color = 'blue')
plt.title('Original')

plt.subplot(1, 2, 2)
plt.hist(log_data, bins = 50, color = 'red')
plt.title('Log Transformed')

plt.show()