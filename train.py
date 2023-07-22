import numpy as np
import tensorflow as tf

# standard resnet block
def residual_block(x, filters, projection):
    x_skip = x 

    # layer 1
    if projection:
        x = tf.keras.layers.Conv2D(filters, (3, 3), (2,2), padding='same', kernel_initializer='he_normal')(x)
    else:
        x = tf.keras.layers.Conv2D(filters, (3, 3), (1,1), padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    # layer 2
    x = tf.keras.layers.Conv2D(filters, (3, 3), (1,1), padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    # addition
    if projection:
        x_skip = tf.keras.layers.Conv2D(filters, (1, 1), (2,2), padding='valid', kernel_initializer='he_normal')(x_skip)
    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)

    return x

# configured for resnet-20
def resnet20(shape_in, classes):
    # input and initial convolution
    x_in = tf.keras.layers.Input(shape_in)
    x = tf.keras.layers.Conv2D(16, (3, 3), (1,1), padding='same', kernel_initializer='he_normal')(x_in)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    # residual blocks
    filters = [16, 32, 64]
    for i in range(3):
        for j in range(3):
            if i > 0 and j == 0:
                x = residual_block(x, filters[i], projection=True)
            else:
                x = residual_block(x, filters[i], projection=False)

    # final dense layer and model 
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x_out = tf.keras.layers.Dense(classes, kernel_initializer='he_normal')(x)
    model = tf.keras.models.Model(x_in, x_out, name='ResNet20')

    return model

if __name__ == "__main__":
    # load dataset
    dataset = np.load('pathmnist.npz')
    keys = list(dataset.keys())
    for key in keys:
        print(key, dataset[key].shape)
    
    # compile model
    model = resnet20((28,28,3), 9)
    optim = tf.keras.optimizers.Adam(lr=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optim, loss=loss, metrics=['accuracy'])
    model.summary()
    