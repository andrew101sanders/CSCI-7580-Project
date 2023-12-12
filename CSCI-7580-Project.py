'''
    CSCI-7580 Project Script

    Paper Title:
        Leveraging DeepSpeed's Hybrid Parallel Programming Model for Efficient Training of Convolutional Neural Networks in Image Classification Tasks

    Authors:
        Andrew Sanders
        Brad Boswell

    Course:
        CSCI-7580 Computer Architecture
    
    Professor:
        Dr. Abdullah Al-Mamun

    Description:
        This file contains the code to train a wide range of convolutional network configurations to test 
        the capabilities of Microsoft's DeepSpeed distributed machine learning framework. To run this, 
        DeepSpeed needs to be installed and a hostfile needs to be specified. The command to run this is

        ```
        deepspeed --hostfile=hostfile CSCI-7580-Project.py --deepspeed
        ```

        which will use DeepSpeed to run this file. If the hostfile has multiple nodes specified, the models will be
        trained using distributed computing.

        An example hostfile is as follows:

        ```
        Andrew slots=1
        Brad slots=1
        ```

        in which Andrew and Brad are specified as Hosts in .ssh/config. IPs can be used instead, but ports need to be
        allowed through the firewall.

        Additionally, a ".deepspeed_env" file can be placed in the home directory (e.g., /home/andrew/) that specifies a few NCCL exports.
        For our tests, we used

        ```
        NCCL_IB_DISABLE=1
        NCCL_SOCKET_IFNAME=wg0
        NCCL_DEBUG=INFO
        ```

        which indicates that InfiniBand should be disabled, the wg0 (WireGuard) interface should be used, and debug level should be INFO
'''


# %%
import deepspeed
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score
import torch.nn.functional as F
import csv
import subprocess
import psutil
from multiprocessing import Process, set_start_method
import sys
import os

# %%
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# %%
# Configuration

# Download training data from open datasets and correctly transform for architectures.
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the images to 224x224, which is a size the architectures were designed for. Based on ImageNet dimensions.
    transforms.ToTensor(),
])

# Benchmark Datasets
# Lambda is used to allow for creation of dataset at runtime
datasets_list = [
    # (
    #  "FashionMNIST",
    #  datasets.FashionMNIST(root="/CSCI-7580-Project/data/", train=True, download=True, transform=ToTensor()),
    #  datasets.FashionMNIST(root="/CSCI-7580-Project/data/", train=False, download=True, transform=ToTensor())
    # ),
    (
     "CIFAR10",
     lambda: (datasets.CIFAR10(root="/CSCI-7580-Project/data/", train=True, download=True, transform=transform),
     datasets.CIFAR10(root="/CSCI-7580-Project/data/", train=False, download=True, transform=transform))
    ),
    (
     "CIFAR100",
     lambda: (datasets.CIFAR100(root="/CSCI-7580-Project/data/", train=True, download=True, transform=transform),
     datasets.CIFAR100(root="/CSCI-7580-Project/data/", train=False, download=True, transform=transform))
    ),
    # (
    #  "MNIST",
    #  datasets.MNIST(root="/CSCI-7580-Project/data/", train=True, download=True, transform=ToTensor()),
    #  datasets.MNIST(root="/CSCI-7580-Project/data/", train=False, download=True, transform=ToTensor())
    # ),
    (
     "SVHN",
     lambda: (datasets.SVHN(root="/CSCI-7580-Project/data/", split="train", download=True, transform=transform),
     datasets.SVHN(root="/CSCI-7580-Project/data/", split="test", download=True, transform=transform))
    )
]

# Benchmark ML architectures
# Second element of each tuple is the "pointer" to the architecture, allowing for creation of model at runtime.
#   otherwise, training would reuse the same weights
models_list = [
    (
        "AlexNet",
        models.alexnet
    ),
    (
        "Inception", # expects 299x299
        models.inception_v3
    ),
    (
        "ResNet50",
        models.resnet50
    ),
    (
        "VGG16",
        models.vgg16
    )
]

# Loss Functions
loss_fn_list = [
    (
        "CrossEntropy",
        nn.CrossEntropyLoss()
    )
]

# List of batch size values to use for training
batch_sizes_list = [256, 512]

# List of learning values to use for training
lr_list = [0.0005, 0.0015]

# List of epoch values to use for training
epoch_list = [10]

# Create a custom DeepSpeed configuration using the passed-in arguments
# This is required for DeepSpeed (see: deepspeed.initialize(config=ds_config))
# Returns:
#   ds_config with argument values
def create_ds_config(batch_size=64, lr=0.001, zero_stage=3):
    ds_config = {
        "train_batch_size": batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": lr,
                "warmup_num_steps": 1000
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage":zero_stage # Values can be 0 for disabled - 3 for fully enabled (stage 3)
        },
        "zero_allow_untested_optimizer": True,
        "steps_per_print": 10
    }
    return ds_config
# %%
# Monitor functions

# Function to get GPU usage and GPU memory usage using the output of nvidia-smi
# The path is hardcoded to the WSL path so it may need to be changed in your case.
# Returns:
#   GPU usage, GPU memory usage
def get_gpu_usage():
    try:
        nvidia_smi_output = subprocess.check_output("/usr/lib/wsl/lib/nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits", shell=True)
        gpu_utilization, gpu_memory_used = map(int, nvidia_smi_output.decode('utf-8').split(','))
        return gpu_utilization, gpu_memory_used
    except Exception as e:
        print(f"Error getting GPU usage: {e}")
        return 0, 0

# Function to get CPU usage and Memory usage. Uses psutil so it should be platform-independent
def get_cpu_memory_usage():
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    return cpu_usage, memory_usage
# %%
# Train and Test functions

# Uses the dataloader (created from dataset) and the DeepSpeed model_engine to train the current epoch.
# Returns 
#   epoch training time, epoch average cpu usage, epoch average memory usage, epoch average gpu usage, epoch average gpu memory usage
def train(dataloader, model_engine, loss_fn):
    size = len(dataloader.dataset)
    model_engine.train() # informs the model that we are about to train. Unfreezes layers that are frozen during evaluation.
    start_time = time.time()  # Get training start time
    cpu_usages, memory_usages, gpu_usages, gpu_memory_usages = [], [], [], []

    for batch, (X, y) in enumerate(dataloader):
        # Ensure data is on the correct device
        # .half() is used to convert data to half-precision for DeepSpeed
        X, y = X.to(model_engine.device).half(), y.to(model_engine.device)

        # Forward pass using model_engine
        pred = model_engine(X)
        loss = loss_fn(pred, y)

        # Backpropagation and weight update
        model_engine.backward(loss)
        model_engine.step()


        # Gets the training metrics (cpu usage, gpu usage, memory usage, gpu memory usage)
        cpu_usage, memory_usage = get_cpu_memory_usage()
        gpu_usage, gpu_memory_usage = get_gpu_usage()
        cpu_usages.append(cpu_usage)
        memory_usages.append(memory_usage)
        gpu_usages.append(gpu_usage)
        gpu_memory_usages.append(gpu_memory_usage)

        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    training_time = time.time() - start_time  # Calculate the total training time
    avg_cpu_usage = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
    avg_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else 0
    avg_gpu_usage = sum(gpu_usages) / len(gpu_usages) if gpu_usages else 0
    avg_gpu_memory_usage = sum(gpu_memory_usages) / len(gpu_memory_usages) if gpu_memory_usages else 0

    return training_time, avg_cpu_usage, avg_memory_usage, avg_gpu_usage, avg_gpu_memory_usage

# Uses dataloader (from dataset) and model_engine (from DeepSpeed model) to test the trained model
# Returns:
#   model accuracy, model precision, model recall
def test(dataloader, model_engine, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    all_probs = []
    all_preds = []  # List to store all predictions
    all_targets = []

    model_engine.eval() # informs the model that we are about to test. Freezes layers that are unfrozen during training.
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(model_engine.local_rank).half(), y.to(model_engine.local_rank)
            pred = model_engine(X)
            probabilities = F.softmax(pred, dim=1)

            correct += (probabilities.argmax(1) == y).type(torch.float).sum().item()
            all_probs.extend(probabilities.tolist())
            all_preds.extend(probabilities.argmax(1).tolist())  # Store all predictions
            all_targets.extend(y.tolist())

    accuracy = 100 * correct / size
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)

    return accuracy, precision, recall

# Uses the current passed configuration to create the model, dataloader, and deepspeed engine and then trains the model using the configuration
def train_model(stdout_fd, stderr_fd, model_index, dataset_index, batch_size_index, lr_index, zero_stage_index, epochs_index, loss_fn_index):
    
    # Redirect subprocess stdout/stderr to parent stdout/stderr to allow for easy debugging
    sys.stdout = os.fdopen(stdout_fd, 'w')
    sys.stderr = os.fdopen(stderr_fd, 'w')

    # Uses indexes of global configuration lists to create the proper configuration that needs to be trained
    model_name, model_constructor = models_list[model_index]
    dataset_name, dataset_constructor = datasets_list[dataset_index]
    dataset_train, dataset_test = dataset_constructor() # Generate the dataset at runtime
    batch_size = batch_sizes_list[batch_size_index]
    lr = lr_list[lr_index]
    zero_stage = zero_stage_index
    epochs = epoch_list[epochs_index]
    loss_fn = loss_fn_list[loss_fn_index]

    ds_config = create_ds_config(batch_size=batch_size, lr=lr, zero_stage=zero_stage) # Create ds_config that is needed by DeepSpeed for distributed training

    # Write training/testing results to csv for later analysis
    # Because we are writing in append mode, we can stop and start this script.
    with open('training_results.csv', 'a', newline='') as file:  # Open a file in append mode
        writer = csv.writer(file)

        model = model_constructor(weights=None) # Create model at runtime so weights are fresh. We do not want to reuse weights.

        # Initialize DeepSpeed using model and ds_config. This also initializes distributed training if it is enabled
        model_engine, _, _, _ = deepspeed.initialize(args=None, model=model, model_parameters=model.parameters(), config_params=ds_config)


        # Create sampler and dataloader for each dataset so processing will be quicker and consistent
        train_sampler = DistributedSampler(dataset_train)
        test_sampler = DistributedSampler(dataset_test)
        # Training Dataloader        
        training_dataloader = DataLoader(dataset_train,
                                     batch_size=batch_size, 
                                     pin_memory=True,
                                     num_workers=0,
                                     sampler=train_sampler)
        # Testing Dataloader
        testing_dataloader = DataLoader(dataset_test, 
                                    batch_size=batch_size, 
                                    pin_memory=True,
                                    num_workers=0,
                                    sampler=test_sampler)
        
        total_training_time = 0 # Used to accumulate training time. This gets added to.
        dataset_size = len(dataset_train)  # Total number of training samples

        for epoch_progress in range(epochs):
            print(f"Epoch {epoch_progress+1}\n-------------------------------")

            # Necessary for samplers
            train_sampler.set_epoch(epoch_progress)
            test_sampler.set_epoch(epoch_progress)

            # Train model and return metrics
            epoch_time, cpu_usage, memory_usage, gpu_usage, gpu_memory_usage = train(training_dataloader, model_engine, loss_fn[1])

            # Add to total training time
            total_training_time += epoch_time

            # Test model and get CNN metrics
            accuracy, precision, recall = test(testing_dataloader, model_engine, loss_fn[1])

            # Calculate throughput
            throughput = (dataset_size * (epoch_progress + 1)) / total_training_time

            # Write all metrics to CSV
            writer.writerow([model_name, 
                                dataset_name, 
                                batch_size, 
                                lr, 
                                zero_stage, 
                                loss_fn[0], 
                                epochs, 
                                epoch_progress+1, # Slightly easier to process if it goes from 1-10 rather than 0-9. This way, we can do (epoch_progress == epochs)
                                f"{throughput:.2f}", 
                                f"{cpu_usage:.2f}", 
                                f"{memory_usage:.2f}", 
                                f"{gpu_usage:.2f}", 
                                f"{gpu_memory_usage:.2f}", 
                                f"{total_training_time:.2f}", 
                                f"{accuracy:.2f}", 
                                f"{precision:.2f}", 
                                f"{recall:.2f}"])
            
            # Go ahead and write to file so we can see results.
            file.flush()

        # Clear GPU memory
        # This may not actually do anything, we just had issues with GPU memory leaks before.
        del model
        del model_engine
        torch.cuda.empty_cache()

# %%
# Training and Testing Model

# Look at results csv and see what models have already been completed.
# Returns:
#   set of completed configs
def get_completed_configs(csv_file):
    completed_configs = set()
    try:
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if int(row['Epoch Progress']) == int(row['Epochs']):
                    config = (row['Model'], row['Dataset'], int(row['Batch Size']), 
                              float(row['Learning Rate']), int(row['Zero Stage']), 
                              row['Loss Function'], int(row['Epochs']))
                    completed_configs.add(config)
    except FileNotFoundError:
        pass  # File not found, so no configurations are completed
    return completed_configs

if __name__ == '__main__':

    # Necessary for some reason
    set_start_method('spawn')

    # Used for subprocesses to redirect output to.
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    # Used to print overall progress
    total_models = len(models_list) * len(datasets_list) * len(batch_sizes_list) * len(lr_list) * len([0, 3]) * len(epoch_list)
    models_trained = 0
    start_time = time.time()

    csv_file = 'training_results.csv'
    completed_configs = get_completed_configs('training_results.csv')
    models_trained += len(completed_configs)

    print(f"completed configs: {completed_configs}")

    # Check if file already exists. If it doesn't, create file and write header line.
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as file:  # Open a file in append mode
        writer = csv.writer(file)

        # Write the header only if the file did not exist
        if not file_exists:
            writer.writerow(['Model', 'Dataset', 'Batch Size', 'Learning Rate', 'Zero Stage', 'Loss Function', 'Epochs', 'Epoch Progress', 'Throughput', 'CPU Usage', 'Memory Usage', 'GPU Usage', 'GPU Memory Usage', 'Total Training Time', 'Accuracy', 'Precision', 'Recall'])

    # Loop through each possible configuration and train it if it is not in the completed_configs
    for model_index, model_info in enumerate(models_list): 
        model_name, model_constructor = model_info
        print(f"Training {model_name}")

        for dataset_index, dataset in enumerate(datasets_list):
            dataset_name, _ = dataset
            print(f"Dataset: {dataset_name}")

            for batch_size_index, batch_size in enumerate(batch_sizes_list):
                print(f"Batch Size: {batch_size}")

                for lr_index, lr in enumerate(lr_list):
                    print(f"Learning Rate: {lr}")

                    # 0 is disabled, 3 is all enabled
                    for zero_stage_index, zero_stage in enumerate([3, 0]):
                        print(f"Zero Stage: {zero_stage}")

                        for epochs_index, epochs in enumerate(epoch_list):
                            print(f"Epochs: {epochs}")

                            for loss_fn_index, loss_fn in enumerate(loss_fn_list):
                                loss_fn_name, _ = loss_fn
                                print(f"Loss Function: {loss_fn_name}")

                                current_config = (model_name, dataset_name, batch_size, lr, zero_stage_index, loss_fn_name, epochs)
                                print(f'current config: {current_config}')

                                # Skip training model if it already has been trained
                                if current_config in completed_configs:
                                    print(f"Skipping completed configuration: {current_config}")
                                    continue

                                # Create subprocess with configuration arguments
                                # Originally, we just called the function without subprocess, but there seems to be a GPU memory leak with PyTorch that caused the GPU memory to slowly grow as it
                                #   created more and more models.
                                p = Process(target=train_model, args=(stdout_fd, stderr_fd, model_index, dataset_index, batch_size_index, lr_index, zero_stage_index, epochs_index, loss_fn_index))
                                p.start()
                                p.join()
                                
                                # Update progress
                                # Used to estimate and print current progress to console.
                                models_trained += 1
                                elapsed_time = time.time() - start_time
                                avg_time_per_model = elapsed_time / models_trained
                                estimated_time_remaining = avg_time_per_model * (total_models - models_trained)
                                print(f"Progress: {models_trained}/{total_models}. Estimated Time Remaining: {estimated_time_remaining/60:.2f} minutes")


    print("Done!")
