import os
import numpy as np
import pandas as pd

datasets = ['pathmnist', 'octmnist', 'tissuemnist']
models = ['convnet', 'vgg16', 'resnet20']
acc = []
for i in models:
    row = {}
    for j in datasets:
        path = os.path.join(os.getcwd(), 'logs', f'{j}_{i}', 'score.npy')
        row[j] = np.load(path)[1]
    acc.append(row)
df = pd.DataFrame(acc, index=models, columns=datasets)
print(df)