import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from models.resnet.resnet_architecture import ResNet, generate_resnet_architecture, ResNetParams
from utils.data_loader import get_cifar10_dataloaders
from utils.training_utils import train_model
from utils.evaluate_utils import evaluate_model
import uuid


def warm_up_resnet(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    classes: list,
    num_epochs: int,
    device: torch.device,
    params: dict = None
) -> dict:
    """
    Script warm_up para treinar e avaliar uma ResNet com parâmetros específicos no CIFAR-10
    
    Args:
        train_loader: DataLoader para treinamento
        val_loader: DataLoader para validação
        test_loader: DataLoader para teste
        classes: Lista de classes
        num_epochs: Número de épocas para treinamento
        device: Dispositivo para execução
        params: Dicionário com parâmetros da arquitetura
        
    Returns:
        dict: Métricas de avaliação da rede
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cria parâmetros ResNetParams a partir do dicionário
    if params is None:
        resnet_params = ResNetParams()
    elif isinstance(params, dict):
        resnet_params = ResNetParams(**params)
    elif isinstance(params, ResNetParams):
        resnet_params = params
    else:
        raise ValueError(f"Tipo de parâmetros não suportado: {type(params)}")
    
    # Gera a arquitetura ResNet com os parâmetros
    model = generate_resnet_architecture(resnet_params).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Treina o modelo
    train_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
    )

    # Avalia o modelo
    test_metrics = evaluate_model(
        model=model, 
        test_loader=test_loader, 
        criterion=criterion, 
        device=device
    )
    
    results = {
        'experiment_id': str(uuid.uuid4()),
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'model': 'ResNet',
        'resnet_params': params.dict(),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'classes': classes
    }

    results_dir = 'results/resnet_experiments'
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f'experiment_{results["experiment_id"]}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Resultados salvos em: {results_file}")

    #weights_file = os.path.join(results_dir, f'experiment_{results["experiment_id"]}_weights.pt')
    #torch.save(model.state_dict(), weights_file)
    #print(f"Pesos do modelo salvos em: {weights_file}")

    return test_metrics

def specialized_training_resnet(
    model: ResNet, test_loader: torch.utils.data.DataLoader, device: torch.device
) -> None:
    pass
