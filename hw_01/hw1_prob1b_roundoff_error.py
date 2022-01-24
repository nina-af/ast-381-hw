import numpy as np

print('Using double-precision...\n')
eps = 1.0

while 1.0 + eps > 1.0:
    print('epsilon = {0:.60f}'.format(eps))
    eps = eps / 2.0

print('\nUsing single-precision...\n')
eps_float32 = np.float32(1.0)

while np.float32(1.0) + eps_float32 > np.float32(1.0):
    print('epsilon = {0:.60f}'.format(eps_float32))
    eps_float32 = eps_float32 / np.float32(2.0)

