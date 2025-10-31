import os
import json
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from models.mobilenet.mobilenet_architecture import generate_mobilenet_architecture
from utils.training_utils import train_model, get_optimized_scheduler
from utils.evaluate_utils import evaluate_model
from models.mobilenet.mobilenet_architecture import MobilenetParams

def warm_up_mobilenet(
    train_loader,
    val_loader,
    test_loader,
    classes,
    num_epochs=3,
    device=None,
    params=None
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = generate_mobilenet_architecture(params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )

    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    results = {
        'experiment_id': str(uuid.uuid4()),
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'model': 'MobileNet',
        'mobilenet_params': params.dict(),
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'classes': classes
    }

    results_dir = 'results/mobilenet_experiments'
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f'experiment_{results["experiment_id"]}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Resultados salvos em: {results_file}")

    #weights_file = os.path.join(results_dir, f'experiment_{results["experiment_id"]}_weights.pt')
    #torch.save(model.state_dict(), weights_file)
    #print(f"Pesos do modelo salvos em: {weights_file}")

    return test_metrics

def specialized_training(
    train_loader,
    val_loader,
    test_loader,
    classes,
    params=None,
    device=None,
    num_epochs=100,
    batch_size=128,
    learning_rate=0.001,
    weight_decay=1e-4,
    use_mixed_precision=True,
    use_compile=True,
    early_stopping_patience=15,
    save_best_model=True,
    experiment_name="mobilenet_specialized"
):
    """
    Treinamento especializado e robusto para MobileNet com hiperparâmetros otimizados.
    
    Args:
        train_loader: DataLoader para treinamento
        val_loader: DataLoader para validação
        test_loader: DataLoader para teste
        classes: Lista de classes
        params: Parâmetros do MobileNet (MobilenetParams)
        device: Dispositivo (CPU/GPU)
        num_epochs: Número de épocas de treinamento
        batch_size: Tamanho do batch
        learning_rate: Learning rate inicial
        weight_decay: Weight decay para regularização
        use_mixed_precision: Usar mixed precision training
        use_compile: Compilar modelo (PyTorch 2.0+)
        early_stopping_patience: Paciência para early stopping
        save_best_model: Salvar melhor modelo
        experiment_name: Nome do experimento
    
    Returns:
        dict: Métricas finais do modelo treinado
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Iniciando treinamento especializado MobileNet")
    print(f"   • Dispositivo: {device}")
    print(f"   • Épocas: {num_epochs}")
    print(f"   • Learning Rate: {learning_rate}")
    print(f"   • Mixed Precision: {use_mixed_precision}")
    print(f"   • Compile: {use_compile}")
    
    # Gera arquitetura MobileNet
    print(f"\n🏗️  Gerando arquitetura MobileNet...")
    model = generate_mobilenet_architecture(params).to(device)
    
    # Conta parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   • Total de parâmetros: {total_params:,}")
    print(f"   • Parâmetros treináveis: {trainable_params:,}")
    
    # Configuração de otimização especializada para MobileNet
    print(f"\n⚙️  Configurando otimização especializada...")
    
    # Otimizador AdamW com weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler - Cosine Annealing com warmup
    scheduler = get_optimized_scheduler(
        optimizer=optimizer,
        scheduler_type='cosine',
        num_epochs=num_epochs,
        warmup_epochs=5,
        min_lr=1e-6
    )
    
    # Função de perda com label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Treinamento especializado
    print(f"\n🔥 Iniciando treinamento especializado...")
    train_metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        use_mixed_precision=use_mixed_precision,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        early_stopping_patience=early_stopping_patience,
        scheduler=scheduler,
        compile_model=use_compile
    )
    
    # Avaliação final especializada
    print(f"\n📊 Avaliação final especializada...")
    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    # Análise de performance
    print(f"\n📈 Análise de Performance:")
    print(f"   • Top-1 Accuracy: {test_metrics['top1_acc']:.2f}%")
    print(f"   • Top-5 Accuracy: {test_metrics['top5_acc']:.2f}%")
    print(f"   • F1-Score: {test_metrics['f1_macro']:.4f}")
    print(f"   • Parâmetros: {test_metrics['total_params']:.2e}")
    print(f"   • Tempo de Inferência: {test_metrics['avg_inference_time']:.4f}s")
    print(f"   • Memória: {test_metrics['memory_used_mb']:.2f} MB")
    print(f"   • GFLOPs: {test_metrics['gflops']:.4f}")
    
    # Salva resultados especializados
    experiment_id = str(uuid.uuid4())
    results = {
        'experiment_id': experiment_id,
        'experiment_name': experiment_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
        'model': 'MobileNet_Specialized',
        'mobilenet_params': params.dict() if hasattr(params, 'dict') else params,
        'training_config': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'use_mixed_precision': use_mixed_precision,
            'use_compile': use_compile,
            'early_stopping_patience': early_stopping_patience,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingWarmRestarts',
            'criterion': 'CrossEntropyLoss with Label Smoothing'
        },
        'model_info': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': test_metrics['memory_used_mb']
        },
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'classes': classes,
        'device': str(device)
    }
    
    # Cria diretório para resultados especializados
    results_dir = 'results/mobilenet_specialized'
    os.makedirs(results_dir, exist_ok=True)
    
    # Salva resultados
    results_file = os.path.join(results_dir, f'{experiment_name}_{experiment_id}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4, default=str)
    print(f"\n💾 Resultados salvos em: {results_file}")
    
    # Salva modelo se solicitado
    if save_best_model:
        weights_file = os.path.join(results_dir, f'{experiment_name}_{experiment_id}_weights.pt')
        
        # Limpa o state_dict antes de salvar (remove chaves extras)
        clean_state_dict = {}
        for key, value in model.state_dict().items():
            if not any(extra_key in key for extra_key in ['total_ops', 'total_params']):
                clean_state_dict[key] = value
        
        torch.save({
            'model_state_dict': clean_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'params': params.dict() if hasattr(params, 'dict') else params,
            'test_metrics': test_metrics,
            'epoch': len(train_metrics)
        }, weights_file)
        print(f"💾 Modelo salvo em: {weights_file}")
    
    # Log final
    print(f"\n✅ Treinamento especializado concluído!")
    print(f"   • Experimento: {experiment_name}")
    print(f"   • ID: {experiment_id}")
    print(f"   • Melhor Top-1: {test_metrics['top1_acc']:.2f}%")
    print(f"   • Eficiência: {test_metrics['gflops']:.4f} GFLOPs")
    
    return test_metrics
