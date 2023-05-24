#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import TensorDataset
import numpy as np
from PIL import Image
from torch.nn import functional as F
from models.Fed import FedSum

def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1

def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1

def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, transform):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, image


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, transform=None, net=None, mnist_joint=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if (mnist_joint is None):
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs, transform), batch_size=self.args.local_bs, shuffle=True)
        else:
            trainset = None
            for i, rot in enumerate(mnist_joint):
                if (i == 0): trainset = DatasetSplit(dataset, idxs, mnist_joint[0])
                else: trainset = torch.utils.data.ConcatDataset([trainset, DatasetSplit(dataset, idxs, rot)])
            self.ldr_train = DataLoader(trainset, batch_size=self.args.local_bs, shuffle=True)

        self.net = net
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        self.grad_dims = []
        for param in self.net.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.args.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.args.device)
        self.grad_ers = []
        self.overwrite_count = 0
        self.not_overwrite_count = 0

    def train(self, buffer, w_glob_er):
        self.net.train()
        # train and update
        epoch_loss = []
        for iteration in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels, not_aug_inputs) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                not_aug_inputs = not_aug_inputs.to(self.args.device)

                self.net.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels)

                # DER
                if (self.args.algo == 'der'):
                    if not buffer.is_empty():
                        buf_inputs, buf_labels, buf_logits = buffer.get_data(
                            self.args.minibatch_size, transform=None)
                        buf_outputs = self.net(buf_inputs)
                        loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

                loss.backward()

                # A-GEM
                if (self.args.algo == 'agem'):
                    if not buffer.is_empty():
                        store_grad(self.net.parameters, self.grad_xy, self.grad_dims)
                        
                        buf_inputs, buf_labels, _ = buffer.get_data(self.args.minibatch_size)
                        self.net.zero_grad()
                        buf_outputs = self.net.forward(buf_inputs)
                        penalty = self.loss_func(buf_outputs, buf_labels)
                        penalty.backward()
                        store_grad(self.net.parameters, self.grad_er, self.grad_dims)
                        
                        dot_prod = torch.dot(self.grad_xy, self.grad_er)
                        if dot_prod.item() < 0:
                            g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                            overwrite_grad(self.net.parameters, g_tilde, self.grad_dims)
                        else:
                            overwrite_grad(self.net.parameters, self.grad_xy, self.grad_dims)

                if iteration == 0 and (self.args.current_it % self.args.task_epoch == 0):
                    buffer.add_data(examples=not_aug_inputs, labels=labels, logits=log_probs.data)
                store_grad(self.net.parameters, self.grad_xy, self.grad_dims)

                # FedGP
                if w_glob_er is not None:
                    dot_prod = torch.dot(self.grad_xy, w_glob_er)
                    if dot_prod.item() < 0:
                        g_tilde = project(gxy=self.grad_xy, ger=w_glob_er)
                        # self.grad_xy = g_tilde
                        overwrite_grad(self.net.parameters, g_tilde, self.grad_dims)
                        self.overwrite_count += 1
                    else:
                        # overwrite_grad(self.net.parameters, self.grad_xy, self.grad_dims)
                        self.not_overwrite_count += 1
                        pass

                self.optimizer.step()

                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iteration, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return self.net.state_dict(), sum(epoch_loss) / len(epoch_loss)


def GradER(buffer, grad_dims, grad_er, buffer_size, net):
    net.train()
    buf_inputs, buf_labels, _ = buffer.get_data(buffer_size)
    dataset = TensorDataset(buf_inputs, buf_labels)
    loader = DataLoader(dataset, batch_size=200)
    net.zero_grad()
    for x, y in loader:
        buf_outputs = net.forward(x)
        loss_func = nn.CrossEntropyLoss()
        penalty = loss_func(buf_outputs, y)
        penalty.backward()
    store_grad(net.parameters, grad_er, grad_dims)
    return grad_er
