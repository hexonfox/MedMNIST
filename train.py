import numpy as np
import tensorflow as tf

def residual_block(x, filters, projection):
    x_skip = x 

    # layer 1
    if projection:
        x = tf.keras.layers.Conv2D(filters, (3, 3), (2,2), padding='same')(x)
    else:
        x = tf.keras.layers.Conv2D(filters, (3, 3), (1,1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    # layer 2
    x = tf.keras.layers.Conv2D(filters, (3, 3), (1,1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    # addition
    if projection:
        x_skip = tf.keras.layers.Conv2D(filters, (1, 1), (2,2), padding='valid')(x_skip)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)

    return x

# configured for resnet-20
def residual_stack(x):
    filters = [16, 32, 64]

    for i in range(3):
        for j in range(3):
            if i > 0 and j == 0:
                x = residual_block(x, filters[i], projection=True)
            else:
                x = residual_block(x, filters[i], projection=False)

    return x

if __name__ == "__main__":
    # load dataset
    dataset = np.load('pathmnist.npz')
    keys = list(dataset.keys())
    for key in keys:
        print(key, dataset[key].shape)
    