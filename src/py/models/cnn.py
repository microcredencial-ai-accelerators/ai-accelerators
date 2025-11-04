import tensorflow as tf
# https://www.kaggle.com/discussions/questions-and-answers/507264

def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), input_shape=input_shape),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(num_classes),
        tf.keras.layers.Activation(tf.nn.softmax, name="Softmax1")
    ])
    return model
