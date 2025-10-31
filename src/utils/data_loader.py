"""
Carregador de Dados
=================

Este módulo é responsável por carregar e preparar datasets para treinamento.
Funcionalidades:
- Carregamento de datasets (CIFAR-10, MNIST, ImageNet, etc.)
- Preprocessamento de imagens (normalização, redimensionamento)
- Data augmentation (rotação, flip, zoom, etc.)
- Divisão em conjuntos de treino, validação e teste
- Criação de DataLoaders otimizados

Centraliza toda a lógica de manipulação de dados para garantir consistência
entre diferentes experimentos e arquiteturas.
"""
import torch
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_cifar10_dataloaders(n_valid=0.2, batch_size=64, num_workers=0):
    """10 Classes"""
    
    transform_train = transforms.Compose([transforms.ToTensor(),
                                    #transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(10),                                    
                                    transforms.RandomCrop(32, padding=4),   
                                    transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  
    ])
    
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)

    n_train = len(train_data)
    indices = list(range(n_train))
    np.random.shuffle(indices)
    split = int(np.floor(n_valid * n_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # Define samplers for obtaining training and validation
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # Prepare data loaders (combine dataset and sampler)
    trainLoader = torch.utils.data.DataLoader(train_data,
                                            batch_size = batch_size,
                                            sampler = train_sampler,
                                            num_workers = num_workers)

    validLoader = torch.utils.data.DataLoader(train_data,
                                            batch_size = batch_size,
                                            sampler = valid_sampler,
                                            num_workers = num_workers)

    testLoader = torch.utils.data.DataLoader(test_data,
                                            batch_size = batch_size,
                                            num_workers = num_workers)

    classes = train_data.classes
    return trainLoader, validLoader, testLoader, classes
