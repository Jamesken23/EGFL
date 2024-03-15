# EGFL
This repo is a paper on Python implementation: **A vulnerability detection framework with enhanced graph feature learning**

# Datasets
We use a recently-released and large-scale dataset [Qian et al., 2023](https://dl.acm.org/doi/10.1145/3543507.3583367) as our benchmark, which mainly covers six categories of vulnerabilities: reentrancy (RE), timestamp dependence (TD), integer overflow/underflow (OF), delegatecall (DE), block number dependency (BN) and unchecked external call (UC). The benchmark dataset comprises 42,910 real-world smart contracts collected from the Ethereum platform, and was created using rigorous data collection and labeling strategies. 

Further instructions on the dataset can be found on [Smart-Contract-Dataset](https://github.com/Messi-Q/Smart-Contract-Dataset), which is constantly being updated to provide more details.

# Required Packages
> - Python 3.7+
> - Keras 2.3.1
> - numpy 1.19.5
> - scikit-learn 1.3.0
> - tensorflow 1.15.0


# Running
To run the program, please use this command: python main_run.py
