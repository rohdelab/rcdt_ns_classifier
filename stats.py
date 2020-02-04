import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys

basedir = sys.argv[1]

def mean_std_stats(result_dir):
    result_files = [os.path.join(basedir, result_dir, 'run{}.pkl'.format(i)) for i in range(5)]
    results = [pickle.load(open(f, 'rb'))['test_acc'] for f in result_files]
    test_acc_mean = np.mean(results)
    std = np.std(results)
    return test_acc_mean, std


log_samples = np.arange(9)

samples = [2**i for i in log_samples]
stats0 = np.array([mean_std_stats('samples-{}-model-MNISTNet'.format(i)) for i in samples])
plt.errorbar(log_samples, stats0[:, 0], stats0[:, 1], fmt='ro-', label='ShallowCNN', alpha=0.5)

# stats1 = np.array([mean_std_stats('samples-{}-model-VGG'.format(i)) for i in samples])
# plt.errorbar(log_samples, stats1[:, 0], stats1[:, 1], fmt='bo-', label='VGG11', alpha=0.5)

# plt.plot(np.log2(samples), stats0[:, 0], 'ro-', label='ShallowCNN')
# plt.plot(np.log2(samples), stats1[:, 0], 'bo-', label='VGG11')
# 

plt.xticks(log_samples, list(map(str, samples)))

plt.xlabel('#samples per-class')
plt.ylabel('test accuracy')
plt.ylim([0,1])


plt.legend()
plt.savefig('out.pdf')
plt.show()
