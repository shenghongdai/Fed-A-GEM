#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_tasks', type=int, default=10, help='')
    parser.add_argument('--task_epoch', type=int, default=20, help='') # communication per task
    # total communication = num_tasks * task_epoch
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--buffer_size', type=int, default=200, help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int, default=32, help='The batch size of the memory buffer.')
    parser.add_argument('--alpha', type=float, default=1.0, help="for DER: hyper-parameter balances the trade-off between the terms")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0, help="SGD momentum")

    parser.add_argument('--layers', default=40, type=int,
                        help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float,
                        help='dropout probability')

    parser.add_argument('--model', type=str, default='resnet18', help='model name: cnn, resnet18')

    parser.add_argument('--dataset', type=str, default='cifar100', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=100, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')

    parser.add_argument('--booster', action='store_true', help='use Fed-A-GEM or not')
    parser.add_argument('--forgetting', action='store_true', help='print forgetting or not')
    parser.add_argument('--num_exp', type=int, default=5, help='number of experiments')
    parser.add_argument('--algo', type=str, default=None, help='algorithm')
    parser.add_argument('--fed_interval', type=int, default=None, help='less communication')
    parser.add_argument('--booster_interval', type=int, default=1, help='less FedGP')
    parser.add_argument('--mnist_permuted', action='store_true', help='use permuted MNIST or rotated MNIST')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    return args

def run_args():
	global args
	args = args_parser()
	print(args)

run_args()
