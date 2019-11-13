""" Functionality for looading MNIST dataset from Kaggle CSVs
    https://www.kaggle.com/ngbolin/mnist-dataset-digit-recognizer/data """

import numpy as np

def load_mnist(location, train=True, labels_as_vectors=True):
    features, labels = list(), list()
    with open(location, 'r') as f:
        header = f.readline()
        for line in f.readlines():
            data = line.split(',')
            if train:
                label = int(data.pop(0))
                if labels_as_vectors:
                    label_vector = np.zeros(10)
                    label_vector[label] = 1
                    label = label_vector
                labels.append(label)
                # frivolty < symmmetry
            ftrs = np.array(data).astype(int)
            features.append(ftrs)
        # symmetry.
        # nice
    return (features, labels)