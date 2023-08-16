# SGVD
This repo is a paper of python implementation : Utilizing bytecode-based multi-modality fusion framework to improve smart contract vulnerability detection. 

Our main intuition is that the opcode (a sequence modality of bytecode) and control flow graph (a graph modality of bytecode) can complement each other in vulnerability detection in smart contracts. To fill the research gap, we propose the SGVD framework that extracts rich semantic information from the opcode and control flow graph, aiming to improve the performance of vulnerability detection.

# Framework

![image](figs/SGVD.pdf)

As shown in this figure, we provide an overview of the SGVD approach, which comprises the three stages:

  * **Data processing:** We collect the smart contract bytecode from the Ethereum platform. The automated tools are then applied to generate the opcode and CFG, respectively.
  
  * **Feature extraction:** We design the TextConformer to learn the semantic features of the opcode, and introduce the MPNN to extract control flow features from the CFG.
  
  * **Vulnerability detection:** We utilize a self-attention network to fuse the semantic and control flow features, which are then fed into the MLP to approach the final prediction.
  

# Datasets
We use a recently-released, large-scale dataset [Qian et al., 2023](https://dl.acm.org/doi/10.1145/3543507.3583367) as our benchmark, which mainly covers four categories of vulnerabilities: reentrancy (RE), timestamp dependence (TD), integer overflow/underflow (OF), and delegatecall (DE). The benchmark dataset comprises 42,910 real-world smart contracts collected from the Ethereum platform, and was created using rigorous data collection and labeling strategies. 

Further instructions on the dataset can be found on [Smart-Contract-Dataset](https://github.com/Messi-Q/Smart-Contract-Dataset), which is constantly being updated to provide more details.

Following their labeling approach, we expand on two common and serious vulnerabilities: block number dependency (BN) and unchecked external call (UC). In total, 680, 2,242, 1,368, 136, 526, and 920 smart contracts have RE, TD, OF, DE, BN, and UC vulnerabilities, respectively.


# Required Packages
> - Python 3.7+
> - Keras 2.3.1
> - numpy 1.19.5
> - scikit-learn 1.3.0
> - tensorflow 1.15.0


# Running
To run program, please use this command: python main_run.py
