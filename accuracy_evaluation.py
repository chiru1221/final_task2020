import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import entropy
from network import MLP_rep
from dataset import Dataset
from baseline import Stream
tf.keras.backend.set_floatx('float64')

class AE(Stream):
    def __init__(self, batch, epoch, sample_num, model_names):
        super().__init__(batch, epoch, sample_num)
        self.model_names = model_names
    
    def test(self, test_ds, model):
        for x_test, y_test in test_ds:
            logit = model(x_test)
            for s in range(self.sample_num - 1):
                logit = tf.math.add(logit, model(x_test))
            logit /= self.sample_num
            
            self.test_score(y_test, logit)
    
    def run(self):
        train_ds, test_ds = self.dataset.mnist()
        def map_evaluate(model_name):
            model = tf.keras.models.load_model(model_name)
            self.test_score.reset_states()
            self.test(test_ds, model)
            print('name : {0:10s}, score : {1:.4f}'.format(model_name, self.test_score.result()))
        
        list(map(map_evaluate, self.model_names))

if __name__ == '__main__':
    evaluation = AE(128, 50, 10, ['baseline_75', 'pm_v2_0', 'pm_v2_1', 'pm_v2_5'])
    evaluation.run()