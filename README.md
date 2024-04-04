# EGFL
This repo is a paper on Python implementation: **A vulnerability detection framework with enhanced graph feature learning**. This paper designs a new deep learning-based framework, named EGFL, that aims to utilize the enhanced graph learning technique to improve the performance of detecting software vulnerabilities (i.e., smart contract vulnerabilities).

# Datasets
We use a recently-released and large-scale dataset [Qian et al., 2023](https://dl.acm.org/doi/10.1145/3543507.3583367) as our benchmark, which mainly covers six categories of vulnerabilities: reentrancy (RE), timestamp dependence (TD), integer overflow/underflow (OF), delegatecall (DE), block number dependency (BN) and unchecked external call (UC). The benchmark dataset comprises 42,910 real-world smart contracts collected from the Ethereum platform [Thomas et al., 2020](https://ieeexplore.ieee.org/document/9284023), and was created using rigorous data collection and labeling strategies. 

Further instructions on the dataset can be found on [Smart-Contract-Dataset](https://github.com/Messi-Q/Smart-Contract-Dataset), which is constantly being updated to provide more details.

# Required Packages
> - Python 3.7+
> - Keras 2.3.1
> - numpy 1.19.5
> - scikit-learn 1.3.0
> - tensorflow 1.15.0


# The construction tools of bytecode CFG
We use a public tool [BinaryCFGExtractor](https://github.com/Messi-Q/BinaryCFGExtractor) to compile a smart contract bytecode into the opcode and corresponding control flow graph (CFG). This compilation tool is mentioned in the paper [Cross-modality mutual learning
for enhancing smart contract vulnerability detection on bytecode](https://dl.acm.org/doi/10.1145/3543507.3583367) and we provide the source code for this tool.

To construct a CFG of bytecode, you also can use the public tool [evm_cfg_builder](https://github.com/crytic/evm_cfg_builder).

We generally use various graph neural networks to handle the CFG and learn the graph features. To learn the graph features of the bytecode CFG, we primainly adopt the GCN model, and refer to some related Github works, such as [GraphExtractor](https://github.com/Messi-Q/SourceGraphExtractor) and [AME](https://github.com/Messi-Q/AMEVulDetector).


# The construction tools from Solidity source code to Ethereum bytecode
If you do not collect enough Ethereum bytecode as your training dataset, you can use the [Solidity compiler](https://github.com/ethereum/solidity/releases) to compile the Solidity source code into Ethereum bytecode. Besides, you can aslo use the [Bytecode to Opcode Disassembler](https://etherscan.io/opcode-tool) to convert the bytecode into the opcode.

As a supplement, you can employ an online Solidity Compiler [remix](https://remix.ethereum.org/) to compile the Solidity source code into Ethereum bytecode.

We will continue to add and update more details on this work.
