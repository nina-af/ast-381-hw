import numpy as np

x         = 0.1
x_float32 = np.float32(x)

print('x = {0:.60g} (double-precision)'.format(x))
print('x = {0:.60g} (single-precision)'.format(x_float32))
