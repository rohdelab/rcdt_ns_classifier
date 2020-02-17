import numpy as np
from scipy.io import loadmat
import os

def new_index_matrix(max_index, indices_perclass, num_classes, repeat):
    seed = int('{}{}{}'.format(indices_perclass, num_classes, repeat))
    np.random.seed(seed)
    return np.random.randint(0, max_index, (num_classes, indices_perclass))


def resize(X, target_size):
    # Assume batch of grayscale images
    assert len(X.shape) == 3
    if target_size == X.shape[1]:
        return X
    X_resize = []
    for i in range(X.shape[0]):
        im = Image.fromarray(X[i])
        im_resize = im.resize((target_size, target_size))
        X_resize.append(np.asarray(im_resize))
    X_resize = np.stack(X_resize, axis=0)
    assert X_resize.shape[0] == X.shape[0]
    return X_resize


def take_samples(data, labels, index, num_classes):
    assert data.shape[0] == labels.shape[0]
    assert index.shape[0] == num_classes
    indexed_data = []
    new_labels = []
    for i in range(num_classes):
       class_data, class_labels = data[labels == i], labels[labels == i]
       indexed_data.append(class_data[index[i]])
       new_labels.append(class_labels[index[i]])
    return np.concatenate(indexed_data), np.concatenate(new_labels)


def load_data(dataset, num_classes):
    cache_file = os.path.join(dataset, 'customizedAffNIST.npz')
    if os.path.exists(cache_file):
        print('loading data from cache file')
        data = np.load(cache_file)
        return (data['x_train'], data['y_train']), (data['x_test'], data['y_test'])

    print('loading data from mat files')
    x_train, y_train, x_test, y_test = [], [], [], []
    for split in ['training', 'testing']:
        for classidx in range(num_classes):
            datafile = os.path.join(dataset, '{}/dataORG_{}.mat'.format(split, classidx))
            # loadmat(datafile)['xxO'] is of shape (H, W, N)
            data = loadmat(datafile)['xxO'].transpose([2, 0, 1]) # transpose to (N, H, W)
            label = np.zeros(data.shape[0], dtype=np.int64)+classidx
            print('split {} class {} data.shape {}'.format(split, classidx, data.shape))
            if split == 'training':
                x_train.append(data)
                y_train.append(label)
            else:
                x_test.append(data)
                y_test.append(label)
    min_samples = min([x.shape[0] for x in x_train])
    x_train = [x[:min_samples] for x in x_train]
    y_train = [y[:min_samples] for y in y_train]
    x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)
    x_test, y_test = np.concatenate(x_test), np.concatenate(y_test)
    print('x_train.shape {} x_test.shape {}'.format(x_train.shape, x_test.shape))

    x_train = x_train / x_train.max(axis=(1, 2), keepdims=True)
    x_test = x_test / x_test.max(axis=(1, 2), keepdims=True)

    x_train = (x_train * 255.).astype(np.uint8)
    x_test = (x_test * 255.).astype(np.uint8)

    # if args.dataset == 'data705_s3_t10':
    #     x_train, x_test = resize(x_train, args.img_size), resize(x_test, args.img_size)

    np.savez(cache_file, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    return (x_train, y_train), (x_test, y_test)


def load_data_3D(dataset, num_classes):
    (x_train, y_train), (x_test, y_test) = load_data(dataset, num_classes)
    # Convert to 1 channel grayscale
    x_train = x_train.reshape(-1, 1, x_train.shape[1], x_train.shape[2])
    # Convert to 3 channels by replicating
    x_train = np.repeat(x_train, axis=1, repeats=3)

    x_test = x_test.reshape(-1, 1, x_test.shape[1], x_test.shape[2])
    x_test = np.repeat(x_test, axis=1, repeats=3)
    return (x_train, y_train), (x_test, y_test)


def train_val_split(x_train, y_train, indices_perclass, num_classes, repeat):
    max_index = x_train.shape[0]//num_classes
    train_index = new_index_matrix(max_index, indices_perclass, num_classes, repeat)

    val_samples = indices_perclass // 10 # Use 10% for validation
    train_samples = indices_perclass - val_samples

    if val_samples >= 1:
        val_index = train_index[:, -val_samples:]
        x_val, y_val = take_samples(x_train, y_train, val_index, num_classes)
        assert x_val.shape[0] == y_val.shape[0]
        print('validation data shape {}'.format(x_val.shape), end=' ')
    else:
        x_val, y_val = None, None
        print('validation data {}'.format(x_val), end=' ')

    train_sub_index = train_index[:, :train_samples]
    x_train_sub, y_train_sub = take_samples(x_train, y_train, train_sub_index, num_classes)
    print('train data shape {}'.format(x_train_sub.shape))

    if x_val is not None:
        assert x_val.shape[0] + x_train_sub.shape[0] == indices_perclass*num_classes
    else:
        assert x_train_sub.shape[0] == indices_perclass*num_classes


    return (x_train_sub, y_train_sub), (x_val, y_val)
