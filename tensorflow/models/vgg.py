def VGG():
    import tensorflow as tf
    from tensorflow import keras

    model = keras.Sequential()
    model.add(keras.layers.Reshape((28,28,1)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    model.add(keras.layers.Dense(1000, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model