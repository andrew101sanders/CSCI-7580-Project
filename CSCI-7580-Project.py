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

# %%
# Download training data from open datasets.

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
     datasets.CIFAR10(root="data", train=True, download=True, transform=transform),
     datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    ),
    (
     "CIFAR100",
     datasets.CIFAR100(root="data", train=True, download=True, transform=transform),
     datasets.CIFAR100(root="data", train=False, download=True, transform=transform)
    ),
    # (
    #  "MNIST",
    #  datasets.MNIST(root="data", train=True, download=True, transform=ToTensor()),
    #  datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())
    # ),
    (
     "SVHN",
     datasets.SVHN(root="data", split="train", download=True, transform=transform),
     datasets.SVHN(root="data", split="test", download=True, transform=transform)
    )
]


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

models_list = [
    (
        "AlexNet",
        models.alexnet
    ),
    (
        "VGG16",
        models.vgg16
    ),
    (
        "Inception",
        models.inception_v3
    ),
    (
        "ResNet50",
        models.resnet50
    )
]

loss_fn_list = [
    (
        "CrossEntropy",
        nn.CrossEntropyLoss()
    )
]

# %%
# DeepSpeed Configuration

def create_ds_config(batch_size=64, lr=0.001, zero_stage=3):
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "optimizer": {
            "type": "SGD",
            "params": {
                "lr": lr
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

        if batch % 100 == 0:
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
# , auc, auprc


# %%
# Training and Testing Model

deepspeed.logger.setLevel("WARNING")

batch_sizes_list = [64, 128, 256]
lr_list = [0.0005, 0.0015]
epoch_list = [10]

total_models = len(models_list) * len(datasets_list) * len(batch_sizes_list) * len(lr_list) * len([False, True]) * len(epoch_list)
models_trained = 0

start_time = time.time()

with open('training_results.csv', 'w', newline='') as file:  # Open a file in append mode
    writer = csv.writer(file)
    # Write header row
    writer.writerow(['Model',
                     'Dataset',
                     'Batch Size',
                     'Learning Rate',
                     'Zero Stage',
                     'Loss Function',
                     'Epochs',
                     'Epoch Progress',
                     'Throughput',
                     'CPU Usage',
                     'Memory Usage',
                     'GPU Usage',
                     'GPU Memory Usage',
                     'Total Training Time',
                     'Accuracy',
                     'Precision',
                     'Recall'])
    for model_info in models_list: 

        model_name, model_constructor = model_info
        print(f"Training {model[0]}")

        for dataset in datasets_list:

            print(f"Dataset: {dataset[0]}")

            for batch_size in batch_sizes_list:

                print(f"Batch Size: {batch_size}")
                # Training Dataloader
                training_dataloader = DataLoader(dataset[1], batch_size=batch_size)
                # Testing Dataloader
                testing_dataloader = DataLoader(dataset[2], batch_size=batch_size)

                for lr in lr_list:

                    print(f"Learning Rate: {lr}")

                    # 0 is disabled, 3 is all enabled
                    for zero_stage in [0, 3]:

                        print(f"Zero Stage: {zero_stage}")
                        ds_config = create_ds_config(batch_size=batch_size, lr=lr, zero_stage=zero_stage)

                        for epochs in epoch_list:

                            print(f"Epochs: {epochs}")

                            for loss_fn in loss_fn_list:

                                print(f"Loss Function: {loss_fn[0]}")

                                model = model_constructor(weights=None)
                                model_engine, _, _, _ = deepspeed.initialize(args=None, model=model, config_params=ds_config)
                                total_training_time = 0
                                dataset_size = len(dataset[1])  # Total number of training samples

                                for epoch_progress in range(epochs):
                                    
                                    print(f"Epoch {epoch_progress+1}\n-------------------------------")

                                    epoch_time, cpu_usage, memory_usage, gpu_usage, gpu_memory_usage = train(training_dataloader, model_engine, loss_fn[1])
                                    total_training_time += epoch_time
                                    accuracy, precision, recall = test(testing_dataloader, model_engine, loss_fn[1])
                                    throughput = (dataset_size * (epoch_progress + 1)) / total_training_time
                                    writer.writerow([model_name, 
                                                     dataset[0], 
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
                            
                                # Update progress
                                models_trained += 1
                                elapsed_time = time.time() - start_time
                                avg_time_per_model = elapsed_time / models_trained
                                estimated_time_remaining = avg_time_per_model * (total_models - models_trained)

                                print(f"Progress: {models_trained}/{total_models}. Estimated Time Remaining: {estimated_time_remaining/60:.2f} minutes")

                                # Clear GPU memory
                                del model
                                del model_engine
                                torch.cuda.empty_cache()
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


