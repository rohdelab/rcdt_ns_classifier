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
log_perclass_samples = np.arange(15)
perclass_samples = np.power(2, log_perclass_samples)
num_classes = 10

# fig = plt.Figure(figsize=(6, 3))
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
ax0.set_title('Train GFLOPS')
ax1.set_title('Average Test GFLOPS')

# files = {'Proposed': 'nsws.hdf5', 'ResNet18': 'nn_resnet18.hdf5', 'ShallowCNN': 'nn_shallowcnn.hdf5', 'VGG11': 'nn_vgg11.hdf5'}
files = {'Proposed': 'nsws.hdf5', 'ShallowCNN': 'nn_shallowcnn.hdf5', 'ResNet18': 'nn_resnet18.hdf5'}
for label, filename in files.items():
    datafile = os.path.join(result_dir, filename)
    print('=========== {} ==========='.format(label))
    if os.path.exists(datafile):
        with h5py.File(datafile, 'r') as f:
            print(f.keys())
            train_gflops, test_gflops = f['train_gflops'][()], f['test_gflops'][()]
            print(train_gflops)
            print(test_gflops)
            # best_run_index = np.argmax(np.mean(accs, axis=1))
            # best_preds = preds[best_run_index]
            # acc_mean, acc_std = accs_stats(best_preds, y_test)
            # precision, recall, fscore, kappa = precision_recall_fscore_stats(best_preds, y_test)
            # print('& {:.4f} (\pm {:.4f}) & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\'.format(acc_mean, acc_std, precision, recall, fscore, kappa))

            # plt.errorbar(log_perclass_samples[:max_index],
            #              np.mean(accs, axis=1), np.std(accs, axis=1), fmt='o-', label=label)
              
            max_index = train_gflops.shape[0]
            if label == 'Proposed':
                train_gflops = num_classes*perclass_samples[:max_index]*0.005+np.mean(train_gflops, axis=1)
                test_gflops = np.ones(max_index) * 0.005 + np.mean(test_gflops, axis=1)/10000
                ax0.plot(log_perclass_samples[:max_index], train_gflops, color='tab:red', linestyle='solid', marker='o', label=label)
                ax1.plot(log_perclass_samples[:max_index], test_gflops, color='tab:red', linestyle='solid', marker='o', label=label)
            else:
                train_gflops = np.mean(train_gflops, axis=1)
                test_gflops = np.mean(test_gflops, axis=1) / 10000
                ax0.plot(log_perclass_samples[:max_index], train_gflops, color='tab:blue', linestyle='solid', marker='o', label=label)
                ax1.plot(log_perclass_samples[:max_index], test_gflops, color='tab:blue', linestyle='solid', marker='o', label=label)
            print(train_gflops)
            print(test_gflops)
   

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

for ax in [ax0, ax1]:
    # ax.set_xticks(log_perclass_samples[:max_index], list(map(str, perclass_samples[:max_index])), rotation=45)
    ax.set_xticks(log_perclass_samples[:max_index])
    ax.set_xticklabels(list(map(str, perclass_samples[:max_index])))
    ax.tick_params(axis='x', rotation=70)
    
    ax.set_xlabel('No. of training samples (per class)')
    ax.set_ylabel('GFLOPS')
    ax.legend()
    # title = result_dir[result_dir.index('final')+6:]
    # if title.endswith('/'):
    #     title = title[:-1]
    # plt.title(title)
plt.grid(linestyle='-', alpha=0.5)
plt.subplots_adjust(bottom=0.2)
#plt.savefig('{}.pdf'.format(title))
plt.show()
#
