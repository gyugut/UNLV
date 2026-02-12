#!/usr/bin/env python
# coding: utf-8

# # Swin-S for CIFAR-10

# ## Import Libraries

# In[ ]:


from os import path, mkdir

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
# import wandb


# ### Huggingface login

# In[ ]:


# Uncomment out the line below when you need to login to Huggingface
#!huggingface-cli login


# ### Check GPU Availability

# In[ ]:


# get_ipython().system('nvidia-smi')


# In[ ]:


# Set CUDA Device Number
DEVICE_NUM = 0

if torch.cuda.is_available():
    device = torch.device(f"cuda:{DEVICE_NUM}")
else:
    device = torch.device("cpu")
    DEVICE_NUM = -1

print(f"INFO: Using device - {device}")


# ## Load DataSets

# In[ ]:


from src.datasets import (
    ImageNet1K, CIFAR100, CIFAR10, DatasetHolder,
    IMAGENET1KConfig, CIFAR100Config, CIFAR10Config
)


# In[ ]:


DATA_ROOT = path.join(".", "data")

# IMAGENETs = DatasetHolder(
#     config=IMAGENET1KConfig,
#     train=ImageNet1K(
#         root=DATA_ROOT, force_download=False, train=True, transform=IMAGENET1KConfig.augmentation
#     ),
#     valid=ImageNet1K(
#         root=DATA_ROOT, force_download=False, valid=True, transform=IMAGENET1KConfig.resizer
#     ),
#     test=ImageNet1K(
#         root=DATA_ROOT, force_download=False, train=False, transform=IMAGENET1KConfig.resizer
#     )
# )
# IMAGENETs.split_train_attack()
# print(f"INFO: Dataset loaded successfully - {IMAGENETs}")

CIFAR100s = DatasetHolder(
    config=CIFAR100Config,
    train=CIFAR100(
        root=DATA_ROOT, download=True, train=True, transform=CIFAR100Config.augmentation
    ),
    test=CIFAR100(
        root=DATA_ROOT, download=True, train=False, transform=CIFAR100Config.resizer
    )
)
CIFAR100s.split_train_valid()
CIFAR100s.split_train_attack()
print(f"INFO: Dataset loaded successfully - {CIFAR100s}")

CIFAR10s = DatasetHolder(
    config=CIFAR10Config,
    train=CIFAR10(
        root=DATA_ROOT, download=True, train=True, transform=CIFAR10Config.augmentation
    ),
    test=CIFAR10(
        root=DATA_ROOT, download=True, train=False, transform=CIFAR10Config.resizer
    )
)
CIFAR10s.split_train_valid()
CIFAR10s.split_train_attack()
print(f"INFO: Dataset loaded successfully - {CIFAR10s}")


# In[ ]:


CHOSEN_DATASET =  CIFAR10s

train_dataset = CHOSEN_DATASET.train
valid_dataset = CHOSEN_DATASET.valid
test_dataset = CHOSEN_DATASET.test

print(f"INFO: Dataset Size - {CHOSEN_DATASET}")


# ## DataLoader

# In[ ]:


# Set Batch Size
BATCH_SIZE = 512, 512, 512


# In[ ]:


MULTI_PROCESSING = True  # Set False if DataLoader is causing issues

from platform import system
if MULTI_PROCESSING and system() != "Windows":  # Multiprocess data loading is not supported on Windows
    import multiprocessing
    cpu_cores = multiprocessing.cpu_count()
    print(f"INFO: Number of CPU cores - {cpu_cores}")
else:
    cpu_cores = 0
    print("INFO: Using DataLoader without multi-processing.")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE[0], shuffle=True, num_workers=cpu_cores)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE[1], shuffle=False, num_workers=cpu_cores)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE[2], shuffle=False, num_workers=cpu_cores)


# In[ ]:


train_loader.show_sample_grid(**CHOSEN_DATASET.config.norm)


# ## Define Model

# In[ ]:


from src.models import (
    ViTBase, ViTLarge,
    SwinTiny, SwinSmall,
    ConvNeXtTiny, ConvNeXtSmall,
    ResNet50,
)


# In[ ]:


TargetModel = ConvNeXtSmall

# WandB Initialization
# try:
#     wandb.finish()
# except:
#     pass
# project = wandb.init(project="Exp_"+CHOSEN_DATASET.config.name.upper(), name=TargetModel.model_name)

# Initialize Model (automatically loads ImageNet-1K pretrained weights)
TargetModel.dataset_name = CHOSEN_DATASET.config.name
model = TargetModel(image_size=CHOSEN_DATASET.config.size, num_classes=CHOSEN_DATASET.num_classes)
model.to(device)


# ## Training Loop

# In[ ]:


def avg(lst):
    try:
        return sum(lst) / len(lst)
    except ZeroDivisionError:
        return 0


# In[ ]:


# Set Epoch Count & Learning Rate
EPOCHS = CHOSEN_DATASET.config.epoch
LEARNING_RATE = 1e-4, 1e-6
WEIGHT_DECAY = 0.05

criterion = nn.CrossEntropyLoss()
# wandb.watch(model, criterion, log="all", log_freq=10)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE[0], weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LEARNING_RATE[1])


# In[ ]:


train_length, valid_length = map(len, (train_loader, valid_loader))

epochs = tqdm(range(EPOCHS), desc="Running Epochs")
with (tqdm(total=train_length, desc="Training") as train_progress,
    tqdm(total=valid_length, desc="Validation") as valid_progress):  # Set up Progress Bars

    for epoch in epochs:
        train_progress.reset(total=train_length)
        valid_progress.reset(total=valid_length)

        train_acc, train_loss, val_acc, val_loss = [], [], [], []

        # Training
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss.append(loss.item())
            train_acc.append((torch.max(outputs, 1)[1] == targets.data).sum().item() / len(inputs))

            train_progress.update(1)
            # if i != train_length-1: wandb.log({'Acc': avg(train_acc)*100, 'Loss': avg(train_loss)})
            print(f"\rEpoch [{epoch+1:4}/{EPOCHS:4}], Step [{i+1:4}/{train_length:4}], Acc: {avg(train_acc):.6%}, Loss: {avg(train_loss):.6f}", end="")

        # Validation
        model.eval()
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)  # but not use model loss

                val_loss.append(criterion(outputs, targets).item())
                val_acc.append((torch.max(outputs, 1)[1] == targets.data).sum().item() / len(inputs))
                valid_progress.update(1)

        # wandb.log({'Train Acc': avg(train_acc)*100, 'Train Loss': avg(train_loss), 'Val Acc': avg(val_acc)*100, 'Val Loss': avg(val_loss)})
        print(f"\rEpoch [{epoch+1:4}/{EPOCHS:4}], Step [{train_length:4}/{train_length:4}], Acc: {avg(train_acc):.6%}, Loss: {avg(train_loss):.6f}, Valid Acc: {avg(val_acc):.6%}, Valid Loss: {avg(val_loss):.6f}", end="\n" if (epoch+1) % 5 == 0 or (epoch+1) == EPOCHS else "")


# In[ ]:


# Model Save
if ADDITIONAL_GPU:
    model.module.save()
else:
    model.save()


# # Model Evaluation

# In[ ]:


# Load Model
model = TargetModel(image_size=CHOSEN_DATASET.config.size, num_classes=len(train_dataset.classes))
# Note: Model is already loaded with ImageNet-1K pretrained weights
# To load your fine-tuned weights, use: model.load("path/to/checkpoint.pth")
model.to(device)


# In[ ]:


corrects = 0
test_length = len(test_dataset)

model.eval()
with torch.no_grad():
    for inputs, targets in tqdm(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        corrects += (preds == targets.data).sum()
        print(f"Model Accuracy: {corrects/test_length:%}", end="\r")


# In[ ]:




