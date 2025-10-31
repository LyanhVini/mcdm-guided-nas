import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import uuid
from datetime import datetime
from models.cnn.cnn_architecture import generate_cnn_architecture, GenericCNN
from utils.training_utils import train_model
from utils.evaluate_utils import evaluate_model


def warm_up_cnn(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    classes: list,
    num_epochs: int,
    device: torch.device,
    params=None
) -> dict:
    """
    Script warm_up para treinar e avaliar uma CNN genérica no CIFAR-10.

    Args:
        train_loader: DataLoader para o conjunto de treinamento.
        val_loader: DataLoader para o conjunto de validação.
        test_loader: DataLoader para o conjunto de teste.
        classes: Lista de nomes das classes.
        num_epochs: Número de épocas para treinamento.
        device: Dispositivo para treinamento (CPU ou CUDA).
    """
    print("\n--- Iniciando warm-up da CNN Genérica ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gera uma nova instância da CNN com base nos parâmetros
    # Assume que generate_cnn_architecture está acessível.
    try:
        model = generate_cnn_architecture(params).to(device)
    except NameError:
        print(
            "Erro: A função 'generate_cnn_architecture' não está definida ou acessível."
        )
        print("Por favor, garanta que ela está importada ou definida no escopo.")
        return  # Impede a continuação se a função não estiver disponível

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Treinamento do modelo
    train_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
    )

    # Avaliação do modelo
    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    # Coleta e salva resultados
    results = {
        "experiment_id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "model_architecture": "GenericCNN",
        "model_parameters": params.model_dump() if hasattr(params, 'model_dump') else params.dict() if hasattr(params, 'dict') else params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "classes": classes,
    }

    results_dir = "results/cnn_experiments"
    os.makedirs(results_dir, exist_ok=True)

    # Salva resultados em JSON
    results_file = os.path.join(
        results_dir, f'experiment_{results["experiment_id"]}.json'
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Resultados salvos em: {results_file}")
    print("--- Warm-up da CNN Genérica Concluído ---")
    
    # Retorna as métricas de teste para uso no otimizador
    return test_metrics


def specialized_training_cnn(
    model: GenericCNN, test_loader: torch.utils.data.DataLoader, device: torch.device
) -> None:
    """
    Função placeholder para treinamento especializado de uma CNN.
    Adicione sua lógica de treinamento especializado aqui, se necessário.
    """
    print("\n--- Iniciando treinamento especializado da CNN (placeholder) ---")
    print("--- Treinamento especializado da CNN Concluído (placeholder) ---")
