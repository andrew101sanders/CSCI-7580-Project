# %% [markdown]
# ### Independent Variables:
# - [x] Dataset (ImageNet, CIFAR, etc.)​
# - [x] Model Architectures​
# - [x] Model Parameters​ ///// AS
# - [ ] GPU Clock Speed​
# - [ ] GPUs Available​
# - [ ] CPU Clock Speed​
# - [ ] CPUs Available​
# - [ ] Shared Memory Resources​
# - [x] ZeRO Algorithm Enabling ///// AS
# 
# ### Dependent Variables:
# - [x] Training Time​ ///// BB
# - [x] Throughput (Samples/Second)​ ///// BB
# - [ ] Resource Utilization​ (nvidia-smi maybe with watch, idk about cpu, idk about ram, maybe htop/top has something to help with this)
# - [ ] CNN Performance ///// BB
#     - [x] Accuracy
#     - [ ] F1-Score
#     - [x] Precision
#     - [x] Recall
#     - [ ] AUC
#     - [ ] AUPRC

# %%
import deepspeed
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score
import torch.nn.functional as F
import csv
import subprocess
import psutil
from multiprocessing import Process, set_start_method
import sys
import os

torch.backends.cudnn.enabled = True

# deepspeed.logger.setLevel("WARNING")

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
    transforms.Resize((224, 224)),  # Resize the images to 224x224
    transforms.ToTensor(),
])

datasets_list = [
    # (
    #  "FashionMNIST",
    #  datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor()),
    #  datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
    # ),
    (
     "CIFAR10",
     lambda: (datasets.CIFAR10(root="data", train=True, download=True, transform=transform),
     datasets.CIFAR10(root="data", train=False, download=True, transform=transform))
    ),
    (
     "CIFAR100",
     lambda: (datasets.CIFAR100(root="data", train=True, download=True, transform=transform),
     datasets.CIFAR100(root="data", train=False, download=True, transform=transform))
    ),
    # (
    #  "MNIST",
    #  datasets.MNIST(root="data", train=True, download=True, transform=ToTensor()),
    #  datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())
    # ),
    (
     "SVHN",
     lambda: (datasets.SVHN(root="data", split="train", download=True, transform=transform),
     datasets.SVHN(root="data", split="test", download=True, transform=transform))
    )
]

# Benchmark ML architectures
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

batch_sizes_list = [32, 64]
lr_list = [0.0005, 0.0015]
epoch_list = [10]

def create_ds_config(batch_size=64, lr=0.001, zero_stage=3):
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
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
            "stage":zero_stage
        },
        "zero_allow_untested_optimizer": True,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
        "steps_per_print": 10000
    }
    return ds_config
# %%
# Monitor functions

def get_gpu_usage():
    try:
        nvidia_smi_output = subprocess.check_output("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits", shell=True)
        gpu_utilization, gpu_memory_used = map(int, nvidia_smi_output.decode('utf-8').split(','))
        return gpu_utilization, gpu_memory_used
    except Exception as e:
        print(f"Error getting GPU usage: {e}")
        return 0, 0

def get_cpu_memory_usage():
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    return cpu_usage, memory_usage
# %%
# Train and Test functions
def train(dataloader, model_engine, loss_fn):
    size = len(dataloader.dataset)
    model_engine.train()
    start_time = time.time()  # Get training start time
    cpu_usages, memory_usages, gpu_usages, gpu_memory_usages = [], [], [], []

    for batch, (X, y) in enumerate(dataloader):
        # Ensure data is on the correct device
        X, y = X.to(model_engine.local_rank).half(), y.to(model_engine.local_rank)

        # Forward pass using model_engine
        pred = model_engine(X)
        loss = loss_fn(pred, y)

        # Backpropagation and weight update
        model_engine.backward(loss)
        model_engine.step()

        cpu_usage, memory_usage = get_cpu_memory_usage()
        gpu_usage, gpu_memory_usage = get_gpu_usage()
        cpu_usages.append(cpu_usage)
        memory_usages.append(memory_usage)
        gpu_usages.append(gpu_usage)
        gpu_memory_usages.append(gpu_memory_usage)

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    training_time = time.time() - start_time  # Calculate the total training time
    avg_cpu_usage = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
    avg_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else 0
    avg_gpu_usage = sum(gpu_usages) / len(gpu_usages) if gpu_usages else 0
    avg_gpu_memory_usage = sum(gpu_memory_usages) / len(gpu_memory_usages) if gpu_memory_usages else 0

    return training_time, avg_cpu_usage, avg_memory_usage, avg_gpu_usage, avg_gpu_memory_usage

def test(dataloader, model_engine, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    all_probs = []
    all_preds = []  # List to store all predictions
    all_targets = []

    model_engine.eval()
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

def train_model(stdout_fd, stderr_fd, model_index, dataset_index, batch_size_index, lr_index, zero_stage_index, epochs_index, loss_fn_index):
    torch.backends.cudnn.enabled = True
    #deepspeed.logger.setLevel("WARNING")
    sys.stdout = os.fdopen(stdout_fd, 'w')
    sys.stderr = os.fdopen(stderr_fd, 'w')

    model_name, model_constructor = models_list[model_index]
    dataset_name, dataset_constructor = datasets_list[dataset_index]
    dataset_train, dataset_test = dataset_constructor()
    batch_size = batch_sizes_list[batch_size_index]
    lr = lr_list[lr_index]
    zero_stage = zero_stage_index
    epochs = epoch_list[epochs_index]
    loss_fn = loss_fn_list[loss_fn_index]

    # Training Dataloader
    training_dataloader = DataLoader(dataset_train,
                                     batch_size=batch_size, 
                                     pin_memory=True,
                                     num_workers=4)
    # Testing Dataloader
    testing_dataloader = DataLoader(dataset_test, 
                                    batch_size=batch_size, 
                                    pin_memory=True,
                                    num_workers=4)

    ds_config = create_ds_config(batch_size=batch_size, lr=lr, zero_stage=zero_stage)

    with open('training_results.csv', 'a', newline='') as file:  # Open a file in append mode
        writer = csv.writer(file)

        model = model_constructor(weights=None)
        model_engine, _, _, _ = deepspeed.initialize(args=None, model=model, config_params=ds_config)
        total_training_time = 0
        dataset_size = len(dataset_train)  # Total number of training samples

        for epoch_progress in range(epochs):
            
            print(f"Epoch {epoch_progress+1}\n-------------------------------")

            epoch_time, cpu_usage, memory_usage, gpu_usage, gpu_memory_usage = train(training_dataloader, model_engine, loss_fn[1])
            total_training_time += epoch_time
            accuracy, precision, recall = test(testing_dataloader, model_engine, loss_fn[1])
            throughput = (dataset_size * (epoch_progress + 1)) / total_training_time
            writer.writerow([model_name, 
                                dataset_name, 
                                batch_size, 
                                lr, 
                                zero_stage, 
                                loss_fn[0], 
                                epochs, 
                                epoch_progress+1, 
                                f"{throughput:.2f}", 
                                f"{cpu_usage:.2f}", 
                                f"{memory_usage:.2f}", 
                                f"{gpu_usage:.2f}", 
                                f"{gpu_memory_usage:.2f}", 
                                f"{total_training_time:.2f}", 
                                f"{accuracy:.2f}", 
                                f"{precision:.2f}", 
                                f"{recall:.2f}"])
            file.flush()

        # Clear GPU memory
        del model
        del model_engine
        torch.cuda.empty_cache()

# %%
# Training and Testing Model

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

    set_start_method('spawn')

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    total_models = len(models_list) * len(datasets_list) * len(batch_sizes_list) * len(lr_list) * len([0, 3]) * len(epoch_list)
    models_trained = 0

    start_time = time.time()

    csv_file = 'training_results.csv'
    completed_configs = get_completed_configs('training_results.csv')
    models_trained += len(completed_configs)

    print(f"completed configs: {completed_configs}")
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as file:  # Open a file in append mode
        writer = csv.writer(file)

        # Write the header only if the file did not exist
        if not file_exists:
            writer.writerow(['Model', 'Dataset', 'Batch Size', 'Learning Rate', 'Zero Stage', 'Loss Function', 'Epochs', 'Epoch Progress', 'Throughput', 'CPU Usage', 'Memory Usage', 'GPU Usage', 'GPU Memory Usage', 'Total Training Time', 'Accuracy', 'Precision', 'Recall'])

        
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
                    for zero_stage_index, zero_stage in enumerate([0, 3]):
                        print(f"Zero Stage: {zero_stage}")

                        for epochs_index, epochs in enumerate(epoch_list):
                            print(f"Epochs: {epochs}")

                            for loss_fn_index, loss_fn in enumerate(loss_fn_list):
                                loss_fn_name, _ = loss_fn
                                print(f"Loss Function: {loss_fn_name}")

                                current_config = (model_name, dataset_name, batch_size, lr, zero_stage_index, loss_fn_name, epochs)
                                print(f'current config: {current_config}')
                                if current_config in completed_configs:
                                    print(f"Skipping completed configuration: {current_config}")
                                    continue

                                p = Process(target=train_model, args=(stdout_fd, stderr_fd, model_index, dataset_index, batch_size_index, lr_index, zero_stage_index, epochs_index, loss_fn_index))
                                p.start()
                                p.join()
                                
                                # Update progress
                                models_trained += 1
                                elapsed_time = time.time() - start_time
                                avg_time_per_model = elapsed_time / models_trained
                                estimated_time_remaining = avg_time_per_model * (total_models - models_trained)

                                print(f"Progress: {models_trained}/{total_models}. Estimated Time Remaining: {estimated_time_remaining/60:.2f} minutes")


    print("Done!")

# %%
# Saving Model
# model_engine.save_checkpoint
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model State to model.pth")

# %%
# Loading Model

# model = NeuralNetwork().to(device)
# model.load_state_dict(torch.load("model.pth"))

# # %%
# # Inference

# classes = [
#     "T-shirt/top",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle boot",
# ]

# model.eval()
# x, y = test_data[0][0], test_data[0][1]
# with torch.no_grad():
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted: "{predicted}", Actual: "{actual}"')


