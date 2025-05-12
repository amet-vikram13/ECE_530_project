import numpy as np
import random
import sys

import os
import argparse
from dataset import get_dataset, get_handler, get_wa_handler
from torchvision import transforms
import torch
import csv
import time

import query_strategies 
import models
from utils import print_log
# import torch.distributed as dist


############################# Specify the hyperparameters #######################################
 
args_pool = {'mnist':
                { 
                 'n_class':10,
                 'channels':1,
                 'size': 28,
                 'transform_tr': transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 128, 'num_workers': 8},
                 'loader_te_args':{'batch_size': 1024, 'num_workers': 8},
                 'normalize':{'mean': (0.1307,), 'std': (0.3081,)},
                },
            'fashionmnist':
                {
                 'n_class':10,
                'channels':1,
                'size': 28,
                'transform_tr': transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 256, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1024, 'num_workers': 1},
                 'normalize':{'mean': (0.1307,), 'std': (0.3081,)},
                },
            'svhn':
                {
                 'n_class':10,
                'channels':3,
                'size': 32,
                'transform_tr': transforms.Compose([ 
                                    transforms.RandomCrop(size = 32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 128, 'num_workers': 8},
                 'loader_te_args':{'batch_size': 1024, 'num_workers': 8},
                 'normalize':{'mean': (0.4377, 0.4438, 0.4728), 'std': (0.1980, 0.2010, 0.1970)},
                },
            'cifar10':
                {
                 'n_class':10,
                 'channels':3,
                 'size': 32,
                 'transform_tr': transforms.Compose([
                                    transforms.RandomCrop(size = 32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 256, 'num_workers': 8},
                 'loader_te_args':{'batch_size': 512, 'num_workers': 8},
                 'normalize':{'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2470, 0.2435, 0.2616)},
                 },
            'gtsrb': 
               {
                 'n_class':43,
                 'channels':3,
                 'size': 32,
                 'transform_tr': transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.RandomCrop(size = 32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])]),
                 'transform_te': transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])]),
                 'loader_tr_args':{'batch_size': 256, 'num_workers': 8},
                 'loader_te_args':{'batch_size': 1024, 'num_workers': 8},
                 'normalize':{'mean': [0.3337, 0.3064, 0.3171], 'std': [0.2672, 0.2564, 0.2629]},
                },
            'tinyimagenet': 
               {
                'n_class':200,
                'channels':3,
                'size': 64,
                'transform_tr': transforms.Compose([
                                    transforms.RandomCrop(size = 64, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                 'transform_te': transforms.Compose([
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                 'loader_tr_args':{'batch_size': 1024, 'num_workers': 4},
                 'loader_te_args':{'batch_size': 512, 'num_workers': 4},
                 'normalize':{'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
                },
            'cifar100': 
               {
                'n_class':100,
                'channels':3,
                'size': 32,
                'transform_tr': transforms.Compose([
                                transforms.RandomCrop(size = 32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
                'transform_te': transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]),
                'loader_tr_args':{'batch_size': 2048, 'num_workers': 4},
                'loader_te_args':{'batch_size': 512, 'num_workers': 8},
                'normalize':{'mean': (0.5071, 0.4867, 0.4408), 'std': (0.2675, 0.2565, 0.2761)},
                }
        }

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, input_dim, output_dim):
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inputs = torch.randn(num_samples, input_dim)
        self.targets = torch.randint(0, output_dim, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


###############################################################################
###############################################################################

if __name__ == '__main__':
    input_dim = 10
    output_dim = 5
    num_samples = 100
    batch_size = 16
    dummy_dataset = DummyDataset(num_samples, input_dim, output_dim)
    dummy_data_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    final_model = stacking_stagewise_training(
        input_dim=input_dim,
        output_dim=output_dim,
        num_stages=3,
        layers_per_stage=1,
        num_epochs_per_stage=5,
        data_loader=dummy_data_loader,
        learning_rate=0.001,
        device=device
    )

    initial_params = [
        np.random.randn(10, 5), # Example parameter 1 (shape 10x5)
        np.random.randn(5, 1),  # Example parameter 2 (shape 5x1)
    ]

    adam_state = initialize_adam_state(initial_params)

    simulated_grads_adam_step1 = [
        np.random.randn(10, 5) * 0.1, # Simulate some gradients
        np.random.randn(5, 1) * 0.1,
    ]

    updated_params_adam_step1, updated_state_adam_step1 = adam_update(
        initial_params,
        simulated_grads_adam_step1,
        adam_state,
        lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
    )

    simulated_grads_adam_step2 = [
        np.random.randn(10, 5) * 0.1,
        np.random.randn(5, 1) * 0.1,
    ]

    updated_params_adam_step2, updated_state_adam_step2 = adam_update(
        updated_params_adam_step1, # Use output params from previous step
        simulated_grads_adam_step2,
        updated_state_adam_step1, # Use output state from previous step
        lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
    )

    params_adan_initial = [
         np.random.randn(10, 5),
         np.random.randn(5, 1),
    ]


    adan_state = initialize_adan_state(params_adan_initial)

    simulated_grads_adan_step1 = [
        np.random.randn(10, 5) * 0.1,
        np.random.randn(5, 1) * 0.1,
    ]

    updated_params_adan_step1, updated_state_adan_step1 = adan_update(
        params_adan_initial,
        simulated_grads_adan_step1,
        adan_state,
        lr=0.001, beta1=0.02, beta2=0.08, beta3=0.01, epsilon=1e-8, weight_decay=0.02
    )

    simulated_grads_adan_step2 = [
        np.random.randn(10, 5) * 0.1,
        np.random.randn(5, 1) * 0.1,
    ]

    updated_params_adan_step2, updated_state_adan_step2 = adan_update(
        updated_params_adan_step1, # Use output params from previous step
        simulated_grads_adan_step2,
        updated_state_adan_step1, # Use output state from previous step
        lr=0.001, beta1=0.02, beta2=0.08, beta3=0.01, epsilon=1e-8, weight_decay=0.02
    )