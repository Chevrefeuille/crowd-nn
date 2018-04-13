from os import listdir
import random
from model import tools
import numpy as np
from model.constants import LABEL_TABLE
import matplotlib.pyplot as plt
import itertools


def get_datasets(data_path, classes):
    """
    Get the dataset for the given classes
    """
    datasets = {}
    for c in classes:
        class_path = data_path + c +'/'
        class_set = [class_path + f for f in listdir(class_path) if 'threshold' in f]
        datasets[c] = class_set
    return datasets

def get_equal_datasets(data_path, classes):
    """
    Get the dataset for the given classes
    """
    datasets, sizes = {}, []
    for c in classes:
        class_path = data_path + c +'/'
        class_set = [class_path + f for f in listdir(class_path) if 'threshold' in f]
        sizes.append(len(class_set))
        datasets[c] = class_set
    min_size = min(sizes)
    for c in classes:
        shuffled_set = random.sample(datasets[c], len(datasets[c]))
        datasets[c] = shuffled_set[:min_size]
    return datasets

def shuffle_data_set(datasets, train_ratio):
    """
    Randomly partition the datasets into training and testing sets
    """
    train_sets, test_sets = {}, {}
    for c, dataset in datasets.items():
        n = len(dataset)
        n_train = round(train_ratio * n)
        shuffled_set = random.sample(dataset, n)
        train_sets[c] = shuffled_set[:n_train]
        test_sets[c] = shuffled_set[n_train:]
    
    return train_sets, test_sets

def get_data_vectors(datasets):
    x, y = [], []
    tot_max = [-float('inf') for i in range(8)]
    tot_min = [float('inf') for i in range(8)]
    for c, dataset in datasets.items():
        for file_path in dataset:
            data = np.load(file_path)
            data_A, data_B = tools.extract_individual_data(data)
            obs_data = tools.compute_observables(data_A, data_B)
            sample_data = tools.get_sample(obs_data)
            max_data = [max(d) for d in sample_data]
            min_data = [min(d) for d in sample_data]
            for i in range(8):
                if max_data[i] > tot_max[i]:
                    tot_max[i] = max_data[i]
                if min_data[i]< tot_min[i]:
                    tot_min[i] = min_data[i]
            x.append(sample_data)
            y.append(LABEL_TABLE[c])

    for i in range(len(x)):
        for j in range(8):
            x[i][j] = (x[i][j] - tot_min[j]) / (tot_max[j] - tot_min[j])
        x[i] = x[i].flatten()
    rng_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rng_state)
    np.random.shuffle(y)
    return np.array(x), np.array(y)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
            
    
