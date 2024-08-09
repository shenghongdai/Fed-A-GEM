#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import matplotlib
matplotlib.use('Agg')
import copy
from torchvision import datasets, transforms
from utils.sampling import mnist_iid, mnist_noniid, cifar100_iid, cifar100_noniid, cifar100_test_split, cifar10_noniid, cifar10_test_split
from utils.options import args_parser
from utils.rotation import Rotation
from utils.permutation import Permutation
from models.Update_booster import LocalUpdate, DatasetSplit, GradER
from models.Fed import FedAvg, FedSum
from models.test import test_img
from data.datasets import CIFAR100_truncated
from models.Nets import CNNMnist
from models.resnet import ResNet18
# from torchvision.models import resnet18, ResNet18_Weights
# from models.ResNet import resnet18_cbam
from models.myNetwork import network
#from utils.buffer import Buffer
import numpy as np
from pathlib import Path
from utils.buffer import Buffer
import torch

def training(trial, args):
    # load dataset and split users
    if args.dataset == 'mnist':
        args.num_classes = 10
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        mnist_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        if args.iid:
            dict_users = mnist_iid(mnist_train, args.num_users) # dict_users[num_users][num_tasks]
        else:
            dict_users = mnist_noniid(mnist_train, args.num_users)
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
        CIFAR_STD = [0.2673, 0.2564, 0.2762]
        train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
        valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
        cifar100_training = CIFAR100_truncated(root='./data/cifar100/', train=True, download=True, transform=train_transform)
        cifar100_test = CIFAR100_truncated(root='./data/cifar100/', train=False, download=True, transform=valid_transform)
        if args.iid:
            dict_users = cifar100_iid(cifar100_training, args.num_users, args.num_tasks)
        else:
            dict_users = cifar100_noniid(cifar100_training, args.num_users, args.num_tasks)
        dict_users_test = cifar100_test_split(cifar100_test, args.num_users, args.num_tasks)
    elif args.dataset == 'cifar10':
        args.num_classes = 10
        transform = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        cifar10_training = datasets.CIFAR10(root='./data/cifar10/', train=True, download=True, transform=transform)
        cifar10_test = datasets.CIFAR10(root='./data/cifar10/', train=False, download=True, transform=transform)
        # # sample users
        dict_users = cifar10_noniid(cifar10_training, args.num_users, args.num_tasks)
        dict_users_test = cifar10_test_split(cifar10_test, args.num_users, args.num_tasks)
    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'resnet18' and (args.dataset == 'cifar100' or args.dataset == 'cifar10'):
        net_glob = ResNet18(num_classes=args.num_classes).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    
    net_glob.train()
    w_glob = net_glob.state_dict()
    buffers = [Buffer(args.buffer_size, args.device) for _ in range(args.num_users)]
    loss_train, acc_tests, acc_mask_classes_tests, acc_tests_cur = [], [], [], []
    grad_dims = []
    for param in net_glob.parameters():
        grad_dims.append(param.data.numel())

    w_glob_er = None
    test_list = []
    if (args.fed_interval is not None):
        w_list = []
        net_local = copy.deepcopy(net_glob)
    
    results = []
    results_mask = []
    forg_list = []
    forg_mask_list = []

    mnist_joint = None
    if (args.dataset == 'mnist'): 
        test_dataset_list = []
        if (args.algo == 'joint'): mnist_joint = []

    for it in range(args.num_tasks * args.task_epoch): # NOTE Total communication
        args.current_it = it
        net_glob.train()
        loss_locals = []
        w_locals = []
        w_er_locals = []
        if(args.fed_interval is not None): is_fed = ((it % args.fed_interval)==0)
        if (args.dataset == 'cifar100'):
            trainset = cifar100_training
            testset = cifar100_test
        elif (args.dataset == 'cifar10'):
            trainset = cifar10_training
            testset = cifar10_test
        elif (args.dataset == 'mnist'):
            if (it % args.task_epoch == 0):
                if (args.mnist_permuted):
                    trans_mnist_rot = transforms.Compose((transforms.ToTensor(), Permutation()))
                else:
                    trans_mnist_rot = transforms.Compose((Rotation(), transforms.ToTensor()))
                mnist_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist_rot)
                mnist_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist_rot)
                if (args.algo == 'joint'):
                    mnist_joint.append(trans_mnist_rot)
                    mnist_train = datasets.MNIST('../data/mnist/', train=True, download=False)
                test_dataset_list.append(mnist_test)
            trainset = mnist_train
            testset = mnist_test
        else:
            exit('Error: unrecognized dataset')
        for idx in range(args.num_users):
            if (args.dataset == 'cifar100' or args.dataset == 'cifar10'):
                if (args.algo == 'joint'): 
                    idxs = set().union(*dict_users[idx][:(int(it/args.task_epoch)+1)])
                else:
                    idxs = dict_users[idx][int(it/args.task_epoch)]
                    if len(idxs) == 0: continue
            elif (args.dataset == 'mnist'):
                idxs = dict_users[idx]
            else:
                exit('Error: unrecognized dataset')
            if ((args.fed_interval is not None) and (it>0) and (not is_fed)):
                net_local.load_state_dict(w_list[-1][idx])
                local = LocalUpdate(args=args, dataset=trainset, idxs=idxs, net=copy.deepcopy(net_local).to(args.device), mnist_joint=mnist_joint)
            else:
                local = LocalUpdate(args=args, dataset=trainset, idxs=idxs, net=copy.deepcopy(net_glob).to(args.device), mnist_joint=mnist_joint)
            w, loss = local.train(buffers[idx], w_glob_er)
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            # print('over_write:', local.overwrite_count)
            # print('not_over_write:', local.not_overwrite_count)
        
        # update global weights
        if (args.fed_interval is not None): 
            w_list.append(copy.deepcopy(w_locals))
            if len(w_list) > 1:
                w_list.pop(0)
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)

        # Fed-A-GEM
        if (args.booster and ((it+1) % args.booster_interval == 0)):
            if ((args.fed_interval is not None) and (((it+1) % args.fed_interval)!=0)):
                pass
            else:
                grad_er = torch.Tensor(np.sum(grad_dims)).to(args.device)
                for idx in range(args.num_users):
                    bs = args.minibatch_size
                    w_er_local = GradER(buffers[idx], grad_dims, copy.deepcopy(grad_er), bs, net=copy.deepcopy(net_glob).to(args.device))
                    w_er_locals.append(copy.deepcopy(w_er_local))
                w_glob_er = FedSum(w_er_locals)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(it, loss_avg))
        loss_train.append(loss_avg)

       
        # testing
        if (it+1)%args.task_epoch == 0:
            if (args.dataset == 'cifar100' or args.dataset == 'cifar10'):
                net_glob.eval()
                acc_test, acc_mask_classes_test = test_img(net_glob, DatasetSplit(testset, idxs=dict_users_test[int(it/args.task_epoch)][0], transform=None), test_loaders = test_list, args = args)
                print("Class-il Testing accuracy: {:.2f}".format(np.mean(acc_test)))
                print("Task-il Testing accuracy: {:.2f}".format(np.mean(acc_mask_classes_test)))
                # current task acc
                # acc_test_cur, acc_mask_classes_test_cur = test_img(net_glob, DatasetSplit(testset, idxs=dict_users_test[it][0], transform=None), test_loaders = None, args = args)
                # print("Current Class-il Testing accuracy: {:.2f}".format(acc_test_cur))
                # print("Current Task-il Testing accuracy: {:.2f}".format(acc_mask_classes_test_cur))
                acc_tests.append(np.mean(acc_test).item())
                acc_mask_classes_tests.append(np.mean(acc_mask_classes_test).item())
                # acc_tests_cur.append(acc_test_cur.item())
                if args.forgetting:
                    results.append(acc_test)
                    results_mask.append(acc_mask_classes_test)
                    n_tasks = len(results)
                    for i in range(n_tasks - 1):
                        results[i] += [0.0] * (n_tasks - len(results[i]))
                        results_mask[i] += [0.0] * (n_tasks - len(results_mask[i]))
                    maxx = np.max(results, axis=0)
                    maxx_mask = np.max(results_mask, axis=0)
                    li = []
                    li_mask = []
                    for task in range(n_tasks - 1):
                        li.append(maxx[task] - results[-1][task])
                        li_mask.append(maxx_mask[task] - results_mask[-1][task])
                    
                    forg = np.mean(li)
                    forg_mask = np.mean(li_mask)
                    print("Class-il Forgetting Metric: {:.2f}".format(forg))
                    print("Task-il Forgetting Metric: {:.2f}".format(forg_mask))
                    forg_list.append(forg)
                    forg_mask_list.append(forg_mask)

            elif (args.dataset == 'mnist'):
                net_glob.eval()
                acc_test = test_img(net_glob, testset, test_loaders = test_list, args = args)
                print("Domain-il accuracy: {:.2f}".format(np.mean(acc_test)))
                # acc_test_cur = test_img(net_glob, testset, test_loaders = None, args = args)
                # print("Current Class-il Testing accuracy: {:.2f}".format(acc_test_cur))
                acc_tests.append(np.mean(acc_test).item())
                # acc_tests_cur.append(acc_test_cur.item())
                if args.forgetting:
                    results.append(acc_test)
                    n_tasks = len(results)
                    for i in range(n_tasks - 1):
                        results[i] += [0.0] * (n_tasks - len(results[i]))
                    maxx = np.max(results, axis=0)
                    li = []
                    for task in range(n_tasks - 1):
                        li.append(maxx[task] - results[-1][task])
                    
                    forg = np.mean(li)
                    print("Domain-il Forgetting Metric: {:.2f}".format(forg))
                    forg_list.append(forg)
            else:
                exit('Error: unrecognized dataset')

    print('train_loss', np.mean(loss_train))
    # print('average current accuracy', np.mean(acc_tests_cur))

    if (args.booster):
        file_name = f'./save/{args.dataset}/{args.algo}+booster/{args.num_users}/'
    else:
        file_name = f'./save/{args.dataset}/{args.algo}/{args.num_users}/'

    Path(file_name).mkdir(parents=True, exist_ok=True)
    file = open(file_name+f'acc_class_{args.lr}_{args.buffer_size}_{args.num_tasks}_{args.task_epoch}_{args.local_ep}_{args.booster_interval}', 'a')
    file.write(str(acc_tests)+'\n')
    file.write(str(forg_list)+'\n')
    file.close()
    
    if (args.dataset == 'cifar100' or args.dataset == 'cifar10'):
        file = open(file_name+f'acc_task_{args.lr}_{args.buffer_size}_{args.num_tasks}_{args.task_epoch}_{args.local_ep}_{args.booster_interval}', 'a')
        file.write(str(acc_mask_classes_tests)+'\n')
        file.write(str(forg_mask_list)+'\n')
        file.close()
        return np.mean(acc_test), np.mean(acc_mask_classes_test)

    return np.mean(acc_test)
