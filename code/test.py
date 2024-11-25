import numpy as np

print(4 % 1)

a = np.array([[0, 0, i] for i in range(256)])
print(np.std(a, axis=-1))
