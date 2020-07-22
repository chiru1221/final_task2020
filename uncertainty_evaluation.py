import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import entropy
from matplotlib import pyplot as plt
from network import MLP_rep
from dataset import Dataset
from baseline import Stream
tf.keras.backend.set_floatx('float64')

class UE(Stream):
    def __init__(self, batch, epoch, sample_num, model_names):
        super().__init__(batch, epoch, sample_num)
        self.model_names = model_names
    
    def test(self, test_ds, model):
        u = list()
        for x_test, y_test in test_ds:
            logit = model(x_test)
            for s in range(self.sample_num - 1):
                logit = tf.math.add(logit, model(x_test))
            logit /= self.sample_num
            uncertainty = entropy(logit.numpy(), axis=1)
            u.extend(uncertainty.tolist())
        return u
                
    def run(self):
        _, mnist_test_ds = self.dataset.mnist()
        _, fm_test_ds = self.dataset.fashion_mnist()
        def map_evaluate(model_name):
            model = tf.keras.models.load_model(model_name)
            uncertainty = self.test(test_ds, model)
            return uncertainty

        test_ds = mnist_test_ds
        uncertainties = list(map(map_evaluate, self.model_names))
        for uncertainty, name in zip(uncertainties, self.model_names):
            plt.figure()
            plt.hist(uncertainty, bins=50)
            plt.xlabel('entropy')
            plt.xlim(0, 2.5)
            plt.ylim(0, 4000)            
            plt.savefig('mnist_' + name + '.jpg')
        
        test_ds = fm_test_ds
        uncertainties = list(map(map_evaluate, self.model_names))
        for uncertainty, name in zip(uncertainties, self.model_names):
            plt.figure()
            plt.hist(uncertainty, bins=50)
            plt.xlabel('entropy')
            plt.xlim(0, 2.5)
            plt.ylim(0, 1000)            
            plt.savefig('fashion_mnist_' + name + '.jpg')


if __name__ == '__main__':
    evaluation = UE(128, 50, 10, ['baseline_75', 'pm_v2_0', 'pm_v2_1', 'pm_v2_5'])
    evaluation.run()