#main.py
import time

import torch
import numpy as np
import random
from utils.options import args
from booster import training

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # if you are using GPU
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # all output files before starting
    open('result.txt', 'w').close()
    if (args.dataset == 'cifar100' or args.dataset == 'cifar10'):
        acc_list_class = []
        acc_list_task = []
        for i in range(args.num_exp):
            set_seed(args.seed+i)
            print(f'====== current experiment: {i} ======')
            acc_class, acc_task = training(i, args)
            acc_list_class.append(acc_class)
            acc_list_task.append(acc_task)
        print('Class-il Testing accuracy:', acc_list_class)
        print("Class-il Avg: {:.2f}".format(np.mean(acc_list_class)))
        print("Class-il Std: {:.2f}".format(np.std(acc_list_class)))
        print("Task-il Testing accuracy:", acc_list_task)
        print("Task-il Avg: {:.2f}".format(np.mean(acc_list_task)))
        print("Task-il Std: {:.2f}".format(np.std(acc_list_task)))
        with open('result.txt', 'a') as f:
            f.write("Class-il Avg: {:.2f}\n".format(np.mean(acc_list_class)))
            f.write("Class-il Std: {:.2f}\n".format(np.std(acc_list_class)))
            f.write("Task-il Avg: {:.2f}\n".format(np.mean(acc_list_task)))
            f.write("Task-il Std: {:.2f}\n".format(np.std(acc_list_task)))
    else:
        acc_list = []
        for i in range(args.num_exp):
            set_seed(args.seed+i)
            print(f'====== current experiment: {i} ======')
            acc = training(i, args)
            acc_list.append(acc)
        print('Testing accuracy:', acc_list)
        print("Avg: {:.2f}".format(np.mean(acc_list)))
        print("Std: {:.2f}".format(np.std(acc_list)))
        with open('result.txt', 'a') as f:
            f.write("Avg: {:.2f}\n".format(np.mean(acc_list)))
            f.write("Std: {:.2f}\n".format(np.std(acc_list)))

if __name__ == "__main__":
    start_total_time = time.time()
    main()
    end_total_time = time.time()
    print('Total ptime: {:.2f}'.format(end_total_time - start_total_time))
