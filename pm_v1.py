import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import entropy
from network import MLP_rep
from dataset import Dataset
from baseline import Stream
tf.keras.backend.set_floatx('float64')

class PMStream(Stream):
    def __init__(self, batch, epoch, sample_num):
        super().__init__(batch, epoch, sample_num)
    
    def train(self, train_ds, model, epoch):
        if epoch > 40:
            l = 1
        else:
            l = 0
        
        for x_train, y_train in train_ds:
            with tf.GradientTape() as tape:
                logit = model(x_train)
                neg_log_likelyhood = sum(self.loss_fn(y_true=y_train, y_pred=logit))
                kl = sum(model.losses)
                
                uq_all = - tf.math.reduce_sum(tf.math.log(logit + 1e-6) * logit, axis=1)
                # uq_all = tf.convert_to_tensor(self.uq_fn(logit.numpy(), axis=1), dtype=tf.float32)
                miss_idx = tf.equal(tf.dtypes.cast(tf.argmax(logit, axis=1), tf.int64), tf.dtypes.cast(tf.argmax(y_train, axis=1), tf.int64))
                # miss_idx = 1.0 - tf.cast(tf.equal(tf.dtypes.cast(tf.argmax(logit, axis=1), tf.int64), tf.dtypes.cast(tf.argmax(y_train, axis=1), tf.int64)), tf.float32)
                # miss_idx = np.array([i for i, (pred, true) in enumerate(zip(np.argmax(logit.numpy(), axis=1), np.argmax(y_train.numpy(), axis=1))) if pred != true])
                # print(miss_idx.shape)
                # print(uq_all.shape)
                # print(type(miss_idx))
                uq_loss = tf.math.reduce_mean(tf.boolean_mask(uq_all, mask=miss_idx))
                # uq_loss = tf.tensordot(uq_all, miss_idx, axes=1) / sum(miss_idx)
                
                loss = ((kl + neg_log_likelyhood) / len(x_train)) - l * uq_loss
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            self.train_score(y_train, logit)
            self.loss_score(loss)
    
    def run(self):
        train_ds, test_ds = self.dataset.mnist()
        # model = MLP_rep()
        model = tf.keras.models.load_model('pm_v1')

        def map_epoch(epoch):
            self.train_score.reset_states()
            self.test_score.reset_states()
            self.loss_score.reset_states()
            self.train(train_ds, model, epoch)
            self.test(test_ds, model)
            print('epoch : {0}, train acc : {1:.4f}, train loss : {2:.4f}, test acc : {3:.4f}'.format(epoch, self.train_score.result(), self.loss_score.result(), self.test_score.result()))

        list(map(map_epoch, range(self.epoch)))
        model.save('pm_v1')


if __name__ == '__main__':
    stream = PMStream(128, 50, 25)
    stream.run()
