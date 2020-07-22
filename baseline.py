import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from network import MLP_rep
from dataset import Dataset
tf.keras.backend.set_floatx('float64')

class Stream:
    def __init__(self, batch, epoch, sample_num):
        self.batch = batch
        self.epoch = epoch
        self.sample_num = sample_num
        self.dataset = Dataset(batch)
        self.loss_fn = tf.keras.losses.categorical_crossentropy
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
        self.train_score = tf.keras.metrics.CategoricalAccuracy()
        self.test_score = tf.keras.metrics.CategoricalAccuracy()
        self.loss_score = tf.keras.metrics.Mean()

    def train(self, train_ds, model):
        for x_train, y_train in train_ds:
            with tf.GradientTape() as tape:
                
                logit = model(x_train)
                for s in range(self.sample_num - 1):
                    logit = tf.math.add(logit, model(x_train))
                logit /= self.sample_num

                neg_log_likelyhood = sum(self.loss_fn(y_true=y_train, y_pred=logit))
                kl = sum(model.losses)
                loss = (kl + neg_log_likelyhood) / len(x_train)
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            self.train_score(y_train, logit)
            self.loss_score(loss)
    
    def test(self, test_ds, model):
        for x_test, y_test in test_ds:
            logit = model(x_test)
            
            self.test_score(y_test, logit)

    def run(self):
        train_ds, test_ds = self.dataset.mnist()
        # model = MLP_rep()
        model = tf.keras.models.load_model('baseline')

        def map_epoch(epoch):
            self.train_score.reset_states()
            self.test_score.reset_states()
            self.loss_score.reset_states()
            self.train(train_ds, model)
            self.test(test_ds, model)
            print('epoch : {0}, train acc : {1:.4f}, train loss : {2:.4f}, test acc : {3:.4f}'.format(epoch, self.train_score.result(), self.loss_score.result(), self.test_score.result()))

        list(map(map_epoch, range(self.epoch)))
        # model.save('baseline')
        model.save('baseline_75')


if __name__ == '__main__':
    # stream = Stream(128, 50, 10)
    stream = Stream(128, 25, 10)
    stream.run()
