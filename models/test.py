#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy

# test_loaders = []

def mask_classes(outputs: torch.Tensor, k: int, args) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    N_CLASSES_PER_TASK = args.num_classes // args.num_tasks
    outputs[:, 0:k * N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * N_CLASSES_PER_TASK:
               args.num_tasks * N_CLASSES_PER_TASK] = -float('inf')

def test_img(net_g, datatest, test_loaders, args):
    net_g.eval()
    accs = []
    accs_mask_classes = []
    # testing
    data_loader = DataLoader(datatest, batch_size=args.bs)
    if (test_loaders == None):
        test_loaders = [copy.deepcopy(data_loader)]
    else:
        test_loaders.append(data_loader)
    l = len(test_loaders)
    for k, data_loader in enumerate(test_loaders):
        test_loss = 0
        correct = 0
        correct_mask_classes = 0
        if (args.dataset == 'cifar100' or args.dataset == 'cifar10'):
            for idx, (data, target, _) in enumerate(data_loader):
                with torch.no_grad():

                    if args.gpu != -1:
                        data, target = data.cuda(), target.cuda()
                    log_probs = net_g(data)
                    # sum up batch loss
                    test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
                    # get the index of the max log-probability
                    y_pred = log_probs.data.max(1, keepdim=True)[1]
                    correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

                    if (args.dataset == 'cifar100' or args.dataset == 'cifar10'):
                        mask_classes(log_probs, k, args)
                        y_pred = log_probs.data.max(1, keepdim=True)[1]
                        correct_mask_classes += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        else:
            for idx, (data, target) in enumerate(data_loader):
                with torch.no_grad():

                    if args.gpu != -1:
                        data, target = data.cuda(), target.cuda()
                    log_probs = net_g(data)
                    # sum up batch loss
                    test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
                    # get the index of the max log-probability
                    y_pred = log_probs.data.max(1, keepdim=True)[1]
                    correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

                    if (args.dataset == 'cifar100' or args.dataset == 'cifar10'):
                        mask_classes(log_probs, k, args)
                        y_pred = log_probs.data.max(1, keepdim=True)[1]
                        correct_mask_classes += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()


        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
        
        if args.verbose:
            print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(data_loader.dataset), accuracy))
        
        accs.append(accuracy)
        if (args.dataset == 'cifar100' or args.dataset == 'cifar10'):
            accs_mask_classes.append(correct_mask_classes / len(data_loader.dataset) * 100)

    if (args.dataset == 'cifar100' or args.dataset == 'cifar10'):
        # return np.mean(accs), np.mean(accs_mask_classes)
        return accs, accs_mask_classes
    # return np.mean(accs)
    return accs

