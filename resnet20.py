import numpy as np
dataset = np.load('pathmnist.npz')
keys = list(dataset.keys())
for key in keys:
    print(key, dataset[key].shape)