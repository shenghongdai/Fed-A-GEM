# FedGP

## Setup

+ Use `./main.py` to run experiments. Make sure you have the necessary dependencies installed.
+ Use `--booster` to control the usage of our algorithm.
+ Use `--forgetting` to enable the calculation of the forgetting metric.
+ Each experiemnt is run in 5 different random seeds.

## Algorithms

+ FedAvg: `--algo sgd`
+ FedCurv: `--algo fedcurv`
+ FedProx: `--algo fedprox`
+ A-GEM: `--algo agem`
+ DER: `--algo der`

## Datasets

+ Rotated MNIST (*Domain-IL*): `--dataset mnist`
+ Permuted MNIST (*Domain-IL*): `--dataset mnist --mnist_permuted`
+ Sequential CIFAR-10 (*Class-Il / Task-IL*): `--dataset cifar10`
+ Sequential CIFAR-100 (*Class-Il / Task-IL*): `--dataset cifar100`

## Examples
+ To run FedAvg using Rotated MNIST data and our FedGP algorithm, execute the following command:
```
python3 main.py --algo sgd --dataset mnist --model cnn --num_channels 1 --local_ep 1 --lr 0.01 --num_classes 10 --booster
```
+ To run A-GEM using Sequential CIFAR-10 data and our FedGP algorithm, execute the following command:
```
python3 main.py --algo agem --dataset cifar10 --num_tasks 5 --num_classes 10 --booster
```
+ To run DER using Sequential CIFAR-100 data and our FedGP algorithm, execute the following command:
```
python3 main.py --algo der --dataset cifar100 --num_classes 100 --booster
```
