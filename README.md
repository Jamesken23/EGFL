# SGVD
This repo is a paper of python implementation : Utilizing bytecode-based multi-modality fusion framework to improve smart contract vulnerability detection. 

Our main intuition is that the opcode (a sequence modality of bytecode) and control flow graph (a graph modality of bytecode) can complement each other in vulnerability detection in smart contracts. To fill the research gap, we propose the SGVD framework that extracts rich semantic information from the opcode and control flow graph, aiming to improve the performance of vulnerability detection.

# Datasets
We use a recently-released, large-scale dataset [Liu et.al.](https://github.com/chenpp1881/GPANet) as our benchmark, which mainly covers four categories of vulnerabilities: reentrancy (RE), timestamp dependence (TD), integer overflow/underflow (OF), and delegatecall (DE). The benchmark dataset comprises 42,910 real-world smart contracts collected from the Ethereum platform, and was created using rigorous data collection and labeling strategies. To elaborate, the dataset uses typical vulnerability characteristics to screen out suspicious smart contracts, which are then subjected to human expert verification. Finally, peers with specialized knowledge are tasked with labeling these suspicious smart contracts.
