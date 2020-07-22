import tensorflow as tf
import tensorflow_probability as tfp
tf.keras.backend.floatx()

class Dataset:
    def __init__(self, batch):
        self.batch = batch
    
    def mnist(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train, x_test = x_train.reshape(-1, 28 * 28), x_test.reshape(-1, 28 * 28)
        y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(self.batch)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.batch)
        return train_ds, test_ds
    
    def fashion_mnist(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train, x_test = x_train.reshape(-1, 28 * 28), x_test.reshape(-1, 28 * 28)
        y_train, y_test = tf.one_hot(y_train, depth=10), tf.one_hot(y_test, depth=10)
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(self.batch)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.batch)
        return train_ds, test_ds
