import os
import argparse
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
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Specify model to use', choices=['vgg16', 'resnet20', 'convnet'], required=True)
    parser.add_argument('--dataset', type=str, help='Specify medmnist dataset to use', choices=['pathmnist'], required=True)
    parser.add_argument('--gpu', type=int, help='Specify gpu index to use', required=False)
    args = parser.parse_args()

    # set gpu index if specified 
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'

    # load dataset
    dataset = np.load(f'{args.dataset}.npz')
    
    # compile model
    if args.model == 'resnet20': 
        model = resnet20((28,28,3), 9)
    elif args.model == 'vgg16': 
        model = tf.keras.applications.VGG16(
            include_top=True,
            weights=None,
            input_tensor=tf.keras.layers.Input((32,32,3)),
            classes=9,
            classifier_activation=None,
        )
        dataset = dict(dataset)
        for i in ['train_images', 'val_images', 'test_images']:
            dataset[i] = np.asarray([
                tf.keras.preprocessing.image.img_to_array(
                    tf.keras.preprocessing.image.array_to_img(
                        image, scale=True
                    ).resize((32,32))
                ) for image in dataset[i] 
            ])
            dataset[i] = tf.keras.applications.vgg16.preprocess_input(dataset[i])
    elif args.model == 'convnet': 
        pass
    optim = tf.keras.optimizers.Adam(lr=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optim, loss=loss, metrics=['accuracy'])
    model.summary()

    # train model
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch, lr: lr*0.1 if epoch == 50 or epoch == 75 else lr
    )
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(os.getcwd(), 'logs', f'{args.dataset}_{args.model}'),
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(os.getcwd(), 'models', f'model_{args.dataset}_{args.model}.h5'),
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max',
    )
    model.fit(
        dataset['train_images']/255.0,
        dataset['train_labels'],
        batch_size=128,
        epochs=100,
        verbose=1,
        callbacks=[lr_scheduler, checkpoint, tensorboard],
        validation_data=(dataset['val_images']/255.0, dataset['val_labels']),
        shuffle=True,
    )

    # load and eval best model
    model = tf.keras.models.load_model(
        os.path.join(os.getcwd(), 'models', f'model_{args.dataset}_{args.model}.h5')
    )
    score = model.evaluate(
        dataset['test_images']/255.0,
        dataset['test_labels'],
        batch_size=128,
        verbose=1,
    )
    print(f"Test loss: {score[0]} - Test acc: {score[1]}")
    np.save(os.path.join(os.getcwd(), 'logs', f'{args.dataset}_{args.model}', 'score.npy'), score)