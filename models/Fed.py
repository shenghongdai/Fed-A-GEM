#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def FedSum(w):
    w_avg = copy.deepcopy(w[0])
    for i in range(1, len(w)):
        w_avg = torch.add(w_avg, w[i])
    w_avg = torch.div(w_avg, len(w))
    return w_avg
