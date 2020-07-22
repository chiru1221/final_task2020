import tensorflow as tf
import tensorflow_probability as tfp
tf.keras.backend.floatx()

class MLP_rep(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tfp.layers.DenseReparameterization(400)
        self.fc2 = tfp.layers.DenseReparameterization(400)
        self.fc3 = tfp.layers.DenseReparameterization(10)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.softmax = tf.keras.layers.Activation('softmax')
    
    def call(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x
