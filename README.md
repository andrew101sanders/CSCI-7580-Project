# Leveraging DeepSpeed's Hybrid Parallel Programming Model for Efficient Training of Convolutional Neural Networks in Image Classification Tasks

This is the respository of our Fall 2023 CSCI-7580 Project for Dr. Abdullah Al-Mamun's CSCI-7580 Computer Architecture Course. The main script is CSCI-7580-Project.py. The results are contained in the following files:
- NonDistributed_AndrewComputer_training_results.csv
- Distributed_AndrewComputer_training_results.csv
- Distributed_BradLaptop_training_results.csv 

NonDistributed is the baseline results, in which the main single computer (Experimental Setup 1) is used for training. Distributed is the comparison results, in which the computer and laptop (Experiment Setup 2) are used for training.

We ran DeepSpeed under [WSL2 Ubuntu](https://learn.microsoft.com/en-us/windows/wsl/install) and used the [WireGuard VPN](https://www.wireguard.com/) to establish a network for easy communication. To run our script, install [DeepSpeed](https://github.com/microsoft/DeepSpeed), create a hostfile (see example hostfile) with configured ssh hosts that have DeepSpeed installed, and execute **deepspeed --hostfile=hostfile CSCI-7580-Project.py --deepspeed**

Additional configurations are needed to allow for GPU/CUDA usage, such as installing the CUDA toolkit in your operating system (see online documentation for [CUDA Toolkit for WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html), for example).

An ".deepspeed_env" file can be placed in the home diretory (e.g., /home/andrew/) that specifies exports for DeepSpeed. For our case, we used

```
NCCL_IB_DISABLE=1
NCCL_SOCKET_IFNAME=wg0
NCCL_DEBUG=INFO
```
which indicates that InfiniBand should be disabled, the wg0 (WireGuard) interface should be used, and debug level should be INFO.

The command we used for our final results is as follows:

**deepspeed --master_addr=20.0.0.2 --master_port=29500 --hostfile=hostfile CSCI-7580-Project.py --deepspeed**

We specified the main computer ip address using --master_addr=20.0.0.2 and port using --master_port=29500. This is to enforce the specific host specific configuration.