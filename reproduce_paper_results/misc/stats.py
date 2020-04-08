import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
import h5py
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

result_dir = sys.argv[1]

# fig = plt.Figure(figsize=(6, 3))
fig, ax = plt.subplots(figsize=(5, 4))

def precision_recall_fscore_stats(multipreds, y_test):
    result = []
    for i in range(multipreds.shape[0]):
        precision, recall, f_score, _ = precision_recall_fscore_support(y_test, multipreds[i], average='weighted')
        kappa = cohen_kappa_score(y_test, multipreds[i])
        result.append((precision, recall, f_score, kappa))
    result = np.stack(result, axis=0)
    return np.mean(result, axis=0)

def accs_stats(multipreds, y_test):
    accs = [accuracy_score(y_test, multipreds[i]) for i in range(multipreds.shape[0])]
    return np.mean(accs), np.std(accs)


def mean_std_stats(result_dir):
    result_files = [os.path.join(result_dir, result_dir, 'run{}.pkl'.format(i)) for i in range(5)]
    result_files = [f for f in result_files if os.path.exists(f) ]
    results = [pickle.load(open(f, 'rb'))['test_acc'] for f in result_files]
    test_acc_mean = np.mean(results)
    std = np.std(results)
    return test_acc_mean, std

log_perclass_samples = np.arange(15)
files = {'Proposed': 'nsws.hdf5', 'ResNet18': 'nn_resnet18.hdf5',
         'ShallowCNN': 'nn_shallowcnn.hdf5', 'VGG11': 'nn_vgg11.hdf5'}
files['RCDT+MLP'] = 'nsws_mlp.hdf5'
for label, filename in files.items():
    datafile = os.path.join(result_dir, filename)
    print('=========== {} ==========='.format(label))
    if os.path.exists(datafile):
        with h5py.File(datafile, 'r') as f:
            accs, preds, y_test = f['accs'][()], f['preds'][()], f['y_test'][()]
            print(accs.squeeze())
            max_index = accs.shape[0]
            best_run_index = np.argmax(np.mean(accs, axis=1))
            best_preds = preds[best_run_index]
            acc_mean, acc_std = accs_stats(best_preds, y_test)
            precision, recall, fscore, kappa = precision_recall_fscore_stats(best_preds, y_test)
            print('& {:.4f} (\pm {:.4f}) & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\'.format(acc_mean, acc_std, precision, recall, fscore, kappa))

            plt.errorbar(log_perclass_samples[:max_index],
                         np.mean(accs, axis=1), np.std(accs, axis=1), fmt='o-', label=label)

perclass_samples = [2**i for i in log_perclass_samples]
# stats_shallowcnn = np.array([mean_std_stats('samples-{}-model-MNISTNet'.format(i)) for i in samples])
# max_samples = stats_shallowcnn.shape[0]
# plt.errorbar(log_perclass_samples[:max_samples], stats_shallowcnn[:, 0], stats_shallowcnn[:, 1], fmt='ro-', label='ShallowCNN', alpha=0.5)
#
# stats_vgg = np.array([mean_std_stats('samples-{}-model-VGG'.format(i)) for i in samples])
# plt.errorbar(log_perclass_samples[:max_samples], stats_vgg[:, 0], stats_vgg[:, 1], fmt='bo-', label='VGG11', alpha=0.5)
#
# stats_resnet = np.array([mean_std_stats('samples-{}-model-ResNet'.format(i)) for i in samples])
# plt.errorbar(log_perclass_samples[:max_samples], stats_resnet[:, 0], stats_resnet[:, 1], fmt='go-', label='ResNet18', alpha=0.5)
#
#
# # plt.plot(np.log2(samples), stats0[:, 0], 'ro-', label='ShallowCNN')
# # plt.plot(np.log2(samples), stats1[:, 0], 'bo-', label='VGG11')
# #
#
plt.xticks(log_perclass_samples[:max_index], list(map(str, perclass_samples[:max_index])), rotation=45)

plt.xlabel('No. of training samples (per class)')
plt.ylabel('Test Accuracy')
plt.ylim([0, 1])
title = result_dir[result_dir.index('final')+6:]
if title.endswith('/'):
    title = title[:-1]
plt.title(title)
plt.legend()
plt.grid(linestyle='-', alpha=0.5)
plt.subplots_adjust(bottom=0.2)
plt.savefig('{}.pdf'.format(title))
plt.show()
#
