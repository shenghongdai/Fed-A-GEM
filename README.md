# Buffer-based Gradient Projection for Continual Federated Learning

<h5 align="center">

[Shenghong Dai](https://scholar.google.com/citations?user=GUarSlcAAAAJ&hl=en), [Jy-yong Sohn](), [Yicong Chen](), [S M Iftekharul Alam](), [Ravikumar Balakrishnan](), [Suman Banerjee](), [Nageen Himayat](), [Kangwook Lee]()


[![arxiv](https://img.shields.io/badge/Arxiv-2409.01585-red)](https://arxiv.org/pdf/2409.01585)
[![openreview](https://img.shields.io/badge/OpenReview-Paper-blue)](https://openreview.net/forum?id=Xz5IcOizQ6)


## 🆕 **News**  
- 🎉 Our paper **"Buffer-based Gradient Projection for Continual Federated Learning"** has been **accepted** by **TMLR (Transactions on Machine Learning Research)**!  


## 💡 Abstract

Continual Federated Learning (CFL) is essential for enabling real-world applications where multiple decentralized clients adaptively learn from continuous data streams. A significant challenge in CFL is mitigating catastrophic forgetting, where models lose previously acquired knowledge when learning new information. Existing approaches often face difficulties due to the constraints of device storage capacities and the heterogeneous nature of data distributions among clients. While some CFL algorithms have addressed these challenges, they frequently rely on unrealistic assumptions about the availability of task boundaries (i.e., knowing when new tasks begin). To address these limitations, we introduce Fed-A-GEM, a federated adaptation of the A-GEM method, which employs a buffer-based gradient projection approach. Fed-A-GEM alleviates catastrophic forgetting by leveraging local buffer samples and aggregated buffer gradients, thus preserving knowledge across multiple clients. Our method is combined with existing CFL techniques, enhancing their performance in the CFL context. Our experiments on standard benchmarks show consistent performance improvements across diverse scenarios. For example, in a task-incremental learning scenario using the CIFAR-100 dataset, our method can increase the accuracy by up to 27%.

## 🚀 Running Experiments
Use `main.py` to run experiments. Each experiment is executed with 5 different random seeds to ensure robustness.

Key options:
- `--booster`: Enable our algorithm.
- `--forgetting`: Calculate the forgetting metric.

## ⚙️ Supported Algorithms

| Algorithm | Command |
|-----------|---------|
| FedAvg    | `--algo sgd` |
| FedCurv   | `--algo fedcurv` |
| FedProx   | `--algo fedprox` |
| A-GEM     | `--algo agem` |
| DER       | `--algo der` |


## 📂 Datasets

| Dataset               | Setting                   | Command                           |
|-----------------------|---------------------------|----------------------------------|
| Rotated MNIST        | Domain-IL                 | `--dataset mnist`                 |
| Permuted MNIST       | Domain-IL                 | `--dataset mnist --mnist_permuted`|
| Sequential CIFAR-10  | Class-IL / Task-IL        | `--dataset cifar10`               |
| Sequential CIFAR-100 | Class-IL / Task-IL        | `--dataset cifar100`              |


Example command to run FedAvg with Rotated MNIST data:
```bash
python3 main.py --algo sgd --dataset mnist --model cnn --num_channels 1 --local_ep 1 --lr 0.01 --num_classes 10 --booster
```

Example command to run A-GEM using Sequential CIFAR-10 data:
```bash
python3 main.py --algo agem --dataset cifar10 --num_tasks 5 --num_classes 10 --booster
```

Example command to run DER using Sequential CIFAR-100 data:
```bash
python3 main.py --algo der --dataset cifar100 --num_classes 100 --booster
```


## 🤖 Contact
For any questions or contributions, please reach out to:
- Shenghong Dai: sdai37@wisc.edu

## 📖 Citation
If you find this project helpful, please consider citing our work:
```bibtex
@article{dai2024buffer,
  title={Buffer-based Gradient Projection for Continual Federated Learning},
  author={Dai, Shenghong and Sohn, Jy-yong and Chen, Yicong and Alam, SM and Balakrishnan, Ravikumar and Banerjee, Suman and Himayat, Nageen and Lee, Kangwook},
  journal={arXiv preprint arXiv:2409.01585},
  year={2024}
}
```

