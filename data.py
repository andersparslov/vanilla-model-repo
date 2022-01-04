import torch
import numpy as np
PATH = 'C:\\Users\\snowy\\Documents\\ml_ops\\dtu_mlops\\data\\corruptmnist\\'

def mnist():
    images_train = []
    labels_train = []
    for i in range(5):
        data = np.load(PATH + f'train_{i}.npz')
        images_train.append(data['images'])
        labels_train.append(data['labels'])
    images_train = torch.Tensor(np.concatenate(images_train)).unsqueeze(1)
    labels_train = torch.Tensor(np.concatenate(labels_train)).long()
    train = torch.utils.data.TensorDataset(images_train, labels_train)
    
    data = np.load(PATH + f'test.npz')
    images_test = torch.Tensor(data['images']).unsqueeze(1)
    labels_test = torch.Tensor(data['labels']).long()
    test = torch.utils.data.TensorDataset(images_test, labels_test)
    return train, test
