#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from collections import Counter
from torch import nn
from torch.utils.data.dataloader import default_collate
import copy
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset

class LambdaModule(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)

def label_squeezing_collate_fn(batch):
    x, y = default_collate(batch)
    return x, y.long().squeeze()

def gaussian_intiailize(model, std=.01):
    modules = [m for n, m in model.named_modules() if 'conv' in n or 'fc' in n]
    parameters = [p for m in modules for p in m.parameters()]

    for p in parameters:
        if p.dim() >= 2:
            nn.init.normal(p, std=std)
        else:
            nn.init.constant(p, 0)

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    if num_users == 10:
        num_shards, num_imgs = 20, 3000 #200, 300
    if num_users == 20:
        num_shards, num_imgs = 40, 1500
    if num_users == 100:
        num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar100_iid(dataset, num_users, num_tasks):
    """
    Sample I.I.D. client data from CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = [i for i in range(len(dataset))]
    labels = dataset.target

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    num_tasks = 20
    num_items_per_task = int(len(dataset)/num_tasks) #2500
    num_items = int(num_items_per_task/num_users) #500

    # divide and assign
    for j in range(num_tasks):
        idxs_per_task = copy.deepcopy(idxs[j*num_items_per_task:(j+1)*num_items_per_task])
        for i in range(num_users):          
            chosen_set = np.random.choice(idxs_per_task, num_items, replace=False)
            rand_set = set(chosen_set)
            print([rand_set])
            # from collections import Counter
            # print(sorted(Counter([dataset[i][1] for i in chosen_set])))
            dict_users[i] = np.concatenate((dict_users[i], [rand_set]), axis=0)
            idxs_per_task = list(set(idxs_per_task)-rand_set)
    # print(len(sorted(list(set().union(*dict_users[0][:2])))))
    # print(sorted(dict_users[0][0]))
    # ceshi
    return dict_users

def cifar10_noniid(dataset, num_users, num_tasks):
    """
    Sample Non-I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    for j in range(num_tasks):
        net_dataidx_map = partition_data(dataset, num_users, j, K=int(10/num_tasks), N=int(len(dataset)/num_tasks), alpha=0.3)
    
        for i in range(num_users):
            dict_users[i] = np.concatenate((dict_users[i], [net_dataidx_map[i]]), axis=0)
    return dict_users

def cifar100_noniid(dataset, num_users, num_tasks):
    """
    Sample Non-I.I.D. client data from CIFAR100 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    for j in range(num_tasks):
        net_dataidx_map = partition_data(dataset, num_users, j, K=int(100/num_tasks), N=int(len(dataset)/num_tasks), alpha=0.3)
    
        for i in range(num_users):
            dict_users[i] = np.concatenate((dict_users[i], [net_dataidx_map[i]]), axis=0)
    return dict_users

def partition_data(dataset, n_nets, j, K, N, alpha):
    min_size = 0
    net_dataidx_map = {}

    while min_size < 10:
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(j*K, (j+1)*K):
            # print(dataset.targets)
            # print(k)
            if hasattr(dataset, 'target'):
                idx_k = np.where(np.array(dataset.target) == k)[0]
            else:
                idx_k = np.where(np.array(dataset.targets) == k)[0]
            #print(idx_k)
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = set(idx_batch[j])

    return net_dataidx_map

def visualize(num_clients, num_classes, stats):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('Set1')
    colors = [cmap(i) for i in range(num_classes)]
    labels = [str(i) for i in range(num_classes)]
    data = []
    for i in range(num_clients):
        client_data = [stats[i]["y"][j] for j in range(num_classes)]
        data.append(client_data)
    new_data = []
    for i in range(num_classes):
        new_data.append([lst[i] for lst in data])
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.bar(np.arange(num_clients), new_data[0], label='Class 1')

    for i in range(1, num_classes):
        ax.bar(np.arange(num_clients), new_data[i], bottom=np.sum(new_data[:i], axis=0), label=f'Class {i+1}')


    # fig, ax = plt.subplots()
    # for i in range(num_clients):  
    #     if i == 0:
    #         ax.bar(range(num_classes), data[i], color=colors, bottom=sum(data[:i]), label=f"Client {i}")
    #     else:            
    #         ax.bar(range(num_classes), data[i], bottom=[sum(x) for x in zip(*data[:i])], color=colors, label=f"Client {i}")
    # ax.set_xticks(range(num_classes))
    # ax.set_xticklabels(labels)
    # ax.legend()
    plt.savefig("myplot.png", dpi=300, bbox_inches='tight')

def dirichlet(
    ori_dataset: Dataset,
    num_clients: int,
    num_classes: int,
    alpha: float,
    least_samples: int,
) -> Tuple[List[List[int]], Dict]:
    min_size = 0
    stats = {}
    targets_numpy = np.array(ori_dataset.target, dtype=np.int32)
    idx = [np.where(targets_numpy == i)[0] for i in range(num_classes)]

    while min_size < least_samples:
        data_indices = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            np.random.shuffle(idx[k])
            distrib = np.random.dirichlet(np.repeat(alpha, num_clients))
            distrib = np.array(
                [
                    p * (len(idx_j) < len(targets_numpy) / num_clients)
                    for p, idx_j in zip(distrib, data_indices)
                ]
            )
            distrib = distrib / distrib.sum()
            distrib = (np.cumsum(distrib) * len(idx[k])).astype(int)[:-1]
            data_indices = [
                np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                for idx_j, idx in zip(data_indices, np.split(idx[k], distrib))
            ]
            min_size = min([len(idx_j) for idx_j in data_indices])
    
    for i in range(num_clients):
        stats[i] = {"x": None, "y": None}
        stats[i]["x"] = len(targets_numpy[data_indices[i]])
        stats[i]["y"] = Counter(targets_numpy[data_indices[i]].tolist())

    dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}
    for i in range(num_clients):
        for j in range(10):
            res = []
            for m in range(len(data_indices[i])):
                if (targets_numpy[data_indices[i]][m] // 10) == j:
                    res.append(data_indices[i][m])
            res = set(res)
            dict_users[i] = np.concatenate((dict_users[i], [res]), axis=0)

    visualize(num_clients, num_classes, stats)
    return dict_users

def cifar10_test_split(dataset, num_users, num_tasks):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :return: dict of image index
    """

    # num_tasks = 20

    dict_users = {i: np.array([], dtype='int64') for i in range(num_tasks)}
    idxs = [i for i in range(len(dataset))]
    labels = dataset.targets

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    num_items_per_task = int(len(dataset)/num_tasks) #500

    # divide and assign
    for j in range(num_tasks):
        rand_set = set(idxs[j*num_items_per_task:(j+1)*num_items_per_task])
        # from collections import Counter
        # print(sorted(Counter([dataset[i][1] for i in rand_set])))
        dict_users[j] = np.concatenate((dict_users[j], [rand_set]), axis=0)
    return dict_users

def cifar100_test_split(dataset, num_users, num_tasks):
    """
    Sample I.I.D. client data from CIFAR100 dataset
    :param dataset:
    :return: dict of image index
    """

    # num_tasks = 20

    dict_users = {i: np.array([], dtype='int64') for i in range(num_tasks)}
    idxs = [i for i in range(len(dataset))]
    labels = dataset.target

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    num_items_per_task = int(len(dataset)/num_tasks) #500

    # divide and assign
    for j in range(num_tasks):
        rand_set = set(idxs[j*num_items_per_task:(j+1)*num_items_per_task])
        # from collections import Counter
        # print(sorted(Counter([dataset[i][1] for i in rand_set])))
        dict_users[j] = np.concatenate((dict_users[j], [rand_set]), axis=0)
    return dict_users

if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
