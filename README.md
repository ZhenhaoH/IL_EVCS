# Economical Electric Vehicle Charging Scheduling via Deep Imitation Learning

[![Paper](https://img.shields.io/badge/Paper-IEEE%20TITS-blue)](https://doi.org/10.1109/TITS.2024.3434734)
[![Python 3.11](https://img.shields.io/badge/Python-3.11.8-green)](https://www.python.org/downloads/release/python-3118/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 Introduction

This repository contains the implementation of a behavior cloning algorithm for electric vehicle charging scheduling optimization.

## 🔧 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ZhenhaoH/IL_EVCS.git
    cd IL_EVCS
    ```

2. Create a Conda environment:
    ```bash
    conda create -n evcs python=3.11.8
    conda activate evcs
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Dataset

The `scenarios/` folder contains the datasets used in our paper, making it easy to reproduce our experimental results.

## 🚀 Usage

### 🏋️ Training

To train our model via behavior cloning:

```bash
python train.py
```

### 🔍 Testing

To evaluate the model on the test set:

```bash
python test.py
```

## 📚 Citation

If you use this code or the results in your research, please cite our paper:

```bibtex
@article{huang2024economical,
  author={Huang, Zhenhao and Wang, Jing and Fan, Xuezhong and Yue, Renfeng and Xiang, Cheng and Gao, Shuhua},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Economical Electric Vehicle Charging Scheduling via Deep Imitation Learning}, 
  year={2024},
  volume={25},
  number={11},
  pages={18196-18210},
  doi={10.1109/TITS.2024.3434734}}
```

## 📫 Contact

For questions or issues, please [open an issue](https://github.com/ZhenhaoH/IL_EVCS/issues) on GitHub.

---

<div align="center">
  <a href="https://github.com/ZhenhaoH/IL_EVCS/stargazers">
    <img alt="Stars" src="https://img.shields.io/github/stars/ZhenhaoH/IL_EVCS?style=social">
  </a>
  <a href="https://github.com/ZhenhaoH/IL_EVCS/network/members">
    <img alt="Forks" src="https://img.shields.io/github/forks/ZhenhaoH/IL_EVCS?style=social">
  </a>
</div>