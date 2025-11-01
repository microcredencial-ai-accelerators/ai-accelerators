import tensorflow as tf

def build_fc_model(input_shape=(28, 28, 1), num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(32),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(num_classes),
        tf.keras.layers.Activation(tf.nn.softmax, name="Softmax1")
    ])
    return model