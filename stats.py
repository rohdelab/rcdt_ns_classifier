import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys

basedir = sys.argv[1]

def get_test_acc_mean(result_dir):
    result_files = [os.path.join(basedir, result_dir, 'run{}.pkl'.format(i)) for i in range(5)]
    test_acc_mean = np.mean([pickle.load(open(f, 'rb'))['test_acc'] for f in result_files])
    return test_acc_mean

n_samples_range = [2**i for i in range(11)]
acc0 = [get_test_acc_mean('samples-{}-model-MNISTNet'.format(i)) for i in n_samples_range]
plt.plot(n_samples_range, acc0, 'ro-', label='ShallowCNN')

acc1 = [get_test_acc_mean('samples-{}-model-VGG'.format(i)) for i in n_samples_range]
plt.plot(n_samples_range, acc1, 'ro-', label='VGG11')


plt.legend()
plt.savefig('out.pdf')
plt.show()
