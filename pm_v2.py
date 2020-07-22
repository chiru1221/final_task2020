import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import entropy
from network import MLP_rep
from dataset import Dataset
from baseline import Stream
tf.keras.backend.set_floatx('float64')

class PMStream(Stream):
    def __init__(self, batch, epoch, sample_num, l):
        super().__init__(batch, epoch, sample_num)
        self.l = l
    
    
    def train(self, train_ds, model, epoch):
        for x_train, y_train in train_ds:
            with tf.GradientTape() as tape:
                # sampling
                logit = model(x_train)
                for s in range(self.sample_num - 1):
                    logit = tf.math.add(logit, model(x_train))
                logit /= self.sample_num

                neg_log_likelyhood = sum(self.loss_fn(y_true=y_train, y_pred=logit))
                kl = sum(model.losses)
                
                uq_all = - tf.math.reduce_sum(tf.math.log(logit + 1e-6) * logit, axis=1)
                # i don't know good doing
                # miss_idx = tf.equal(tf.dtypes.cast(tf.argmax(logit, axis=1), tf.int64), tf.dtypes.cast(tf.argmax(y_train, axis=1), tf.int64))
                miss_idx = tf.math.logical_not(tf.equal(tf.dtypes.cast(tf.argmax(logit, axis=1), tf.int64), tf.dtypes.cast(tf.argmax(y_train, axis=1), tf.int64)))
                
                uq_loss = tf.math.reduce_mean(tf.boolean_mask(uq_all, mask=miss_idx))
                if np.isnan(uq_loss):
                    loss = ((kl + neg_log_likelyhood) / len(x_train))
                else:
                    loss = ((kl + neg_log_likelyhood) / len(x_train)) - self.l * uq_loss
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            self.train_score(y_train, logit)
            self.loss_score(loss)
    
    def run(self):
        train_ds, test_ds = self.dataset.mnist()
        model = tf.keras.models.load_model('baseline')
        test_acc = list()

        def map_epoch(epoch):
            self.train_score.reset_states()
            self.test_score.reset_states()
            self.loss_score.reset_states()
            self.train(train_ds, model, epoch)
            self.test(test_ds, model)
            print('epoch : {0}, train acc : {1:.4f}, train loss : {2:.4f}, test acc : {3:.4f}'.format(epoch, self.train_score.result(), self.loss_score.result(), self.test_score.result()))
            test_acc.append(self.test_score.result().numpy())

        list(map(map_epoch, range(self.epoch)))
        model.save('pm_v2_{0}'.format(int(l * 10)))
        np.save('pm_v2_acc_{0}'.format(int(l * 10)), np.array(test_acc))


if __name__ == '__main__':
    ls = [0.05, 0.1, 0.5]
    for l in ls:
        stream = PMStream(128, 25, 10, l)
        stream.run()
