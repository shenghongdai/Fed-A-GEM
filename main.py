# main.py
import time

import torch
import numpy as np
import random
from utils.options import args
from booster import training

# Set random seeds for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # Ensuring deterministic algorithms on GPU
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Main function to execute experiments
def main():
    # Clear the result file before starting
    open('result.txt', 'w').close()
    # Check if the dataset is CIFAR10 or CIFAR100
    if (args.dataset == 'cifar100' or args.dataset == 'cifar10'):
        acc_list_class = []
        acc_list_task = []
        # Run experiments for specified number of times
        for i in range(args.num_exp):
            set_seed(args.seed+i)
            print(f'====== current experiment: {i} ======')
            acc_class, acc_task = training(i, args)
            acc_list_class.append(acc_class)
            acc_list_task.append(acc_task)
        # Display results for Class-IL and Task-IL
        print('Class-il Testing accuracy:', acc_list_class)
        print("Class-il Avg: {:.2f}".format(np.mean(acc_list_class)))
        print("Class-il Std: {:.2f}".format(np.std(acc_list_class)))
        print("Task-il Testing accuracy:", acc_list_task)
        print("Task-il Avg: {:.2f}".format(np.mean(acc_list_task)))
        print("Task-il Std: {:.2f}".format(np.std(acc_list_task)))
        # Save results to file
        with open('result.txt', 'a') as f:
            f.write("Class-il Avg: {:.2f}\n".format(np.mean(acc_list_class)))
            f.write("Class-il Std: {:.2f}\n".format(np.std(acc_list_class)))
            f.write("Task-il Avg: {:.2f}\n".format(np.mean(acc_list_task)))
            f.write("Task-il Std: {:.2f}\n".format(np.std(acc_list_task)))
    else:
        acc_list = []
        # Run experiments for other datasets
        for i in range(args.num_exp):
            set_seed(args.seed+i)
            print(f'====== current experiment: {i} ======')
            acc = training(i, args)
            acc_list.append(acc)
        # Display results
        print('Testing accuracy:', acc_list)
        print("Avg: {:.2f}".format(np.mean(acc_list)))
        print("Std: {:.2f}".format(np.std(acc_list)))
        # Save results to file
        with open('result.txt', 'a') as f:
            f.write("Avg: {:.2f}\n".format(np.mean(acc_list)))
            f.write("Std: {:.2f}\n".format(np.std(acc_list)))

# Entry point of the script
if __name__ == "__main__":
    start_total_time = time.time()
    main()
    end_total_time = time.time()
    print('Total ptime: {:.2f}'.format(end_total_time - start_total_time))
