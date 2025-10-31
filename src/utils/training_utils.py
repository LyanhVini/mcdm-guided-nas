"""
Utilitários de Treinamento
=========================

Funções auxiliares para treinamento de modelos neurais.
Funcionalidades:
- Configuração de otimizadores e schedulers
- Implementação de callbacks (early stopping, learning rate decay)
- Métricas de treinamento e validação
- Data augmentation e preprocessing
- Checkpointing e restauração de modelos

Centraliza todas as funções relacionadas ao treinamento,
garantindo consistência e reprodutibilidade dos experimentos.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import List, Dict, Optional
import time

def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    num_epochs: int, 
    device: torch.device,
    # Novos parâmetros de otimização
    use_mixed_precision: bool = False,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    early_stopping_patience: int = 10,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    compile_model: bool = False
) -> List[Dict[str, float]]:
    """
    Treina um modelo no CIFAR-10 de forma otimizada mantendo simplicidade.

    Args:
        model: Modelo PyTorch (herda de nn.Module).
        train_loader: DataLoader para dados de treinamento.
        val_loader: DataLoader para dados de validação.
        criterion: Função de perda (ex.: CrossEntropyLoss).
        optimizer: Otimizador (ex.: Adam).
        num_epochs: Número de épocas.
        device: Dispositivo (CPU ou GPU).
        use_mixed_precision: Usar mixed precision training (acelera GPU).
        gradient_accumulation_steps: Passos para acumular gradientes.
        max_grad_norm: Valor máximo para gradient clipping.
        early_stopping_patience: Paciência para early stopping.
        scheduler: Learning rate scheduler opcional.
        compile_model: Compilar modelo (PyTorch 2.0+).

    Returns:
        List[Dict[str, float]]: Lista de dicionários com métricas por época
            (train_loss, train_acc, valid_loss, valid_acc).
    """
    model.to(device)
    
    # Compilar modelo para otimização (PyTorch 2.0+)
    if compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("✓ Modelo compilado com torch.compile")
        except Exception as e:
            print(f"⚠ Não foi possível compilar o modelo: {e}")
    
    # Inicializar mixed precision scaler
    scaler = GradScaler() if use_mixed_precision and device.type == 'cuda' else None
    if scaler:
        print("✓ Mixed precision habilitado")
    
    # Variáveis para early stopping
    best_val_acc = 0.0
    patience_counter = 0
    
    metrics = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # === FASE DE TREINAMENTO ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Reshape para modelos MLP/DBN
            if isinstance(model, (nn.Sequential)) or 'MLP' in model.__class__.__name__ or 'DBN' in model.__class__.__name__:
                inputs = inputs.view(inputs.size(0), -1)
            
            # Forward pass com mixed precision
            if scaler:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / gradient_accumulation_steps
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels) / gradient_accumulation_steps
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Estatísticas
            train_loss += loss.item() * inputs.size(0) * gradient_accumulation_steps
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # === FASE DE VALIDAÇÃO ===
        model.eval()
        valid_loss, valid_correct, valid_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                if isinstance(model, (nn.Sequential)) or 'MLP' in model.__class__.__name__ or 'DBN' in model.__class__.__name__:
                    inputs = inputs.view(inputs.size(0), -1)
                
                # Forward pass com mixed precision
                if scaler:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                valid_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                valid_total += labels.size(0)
                valid_correct += predicted.eq(labels).sum().item()
        
        # Atualizar learning rate scheduler
        if scheduler:
            scheduler.step()
        
        # Calcular métricas da época
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss / train_total,
            'train_acc': 100.0 * train_correct / train_total,
            'valid_loss': valid_loss / valid_total,
            'valid_acc': 100.0 * valid_correct / valid_total,
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time': time.time() - epoch_start
        }
        metrics.append(epoch_metrics)
        
        # Early stopping
        if epoch_metrics['valid_acc'] > best_val_acc:
            best_val_acc = epoch_metrics['valid_acc']
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Log do progresso
        print(f"Epoch {epoch + 1}/{num_epochs} ({epoch_metrics['epoch_time']:.1f}s): "
              f"Train Loss: {epoch_metrics['train_loss']:.4f}, "
              f"Train Acc: {epoch_metrics['train_acc']:.2f}%, "
              f"Valid Loss: {epoch_metrics['valid_loss']:.4f}, "
              f"Valid Acc: {epoch_metrics['valid_acc']:.2f}%, "
              f"LR: {epoch_metrics['lr']:.6f}")
        
        # Verificar early stopping
        if patience_counter >= early_stopping_patience:
            print(f"🛑 Early stopping após {patience_counter} épocas sem melhoria")
            break
    
    total_time = time.time() - start_time
    print(f"✓ Treinamento concluído em {total_time:.1f}s")
    print(f"✓ Melhor acurácia de validação: {best_val_acc:.2f}%")
    
    return metrics


def create_optimized_dataloader(dataset, batch_size: int, shuffle: bool = True, 
                              num_workers: int = 4, pin_memory: bool = True) -> DataLoader:
    """
    Cria um DataLoader otimizado para performance.
    
    Args:
        dataset: Dataset PyTorch
        batch_size: Tamanho do batch
        shuffle: Se deve embaralhar os dados
        num_workers: Número de workers para carregamento
        pin_memory: Usar pin memory para GPU
    
    Returns:
        DataLoader otimizado
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2
    )


def get_optimized_scheduler(optimizer, scheduler_type: str = 'cosine', **kwargs):
    """
    Retorna um scheduler otimizado.
    
    Args:
        optimizer: Otimizador PyTorch
        scheduler_type: Tipo de scheduler ('cosine', 'step', 'plateau')
        **kwargs: Argumentos específicos do scheduler
    
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=kwargs.get('factor', 0.1),
            patience=kwargs.get('patience', 5),
            verbose=True
        )
    else:
        raise ValueError(f"Scheduler não suportado: {scheduler_type}")


def train_model_optimized_example(model, train_dataset, val_dataset, num_epochs=100):
    """
    Exemplo de uso completo com todas as otimizações.
    
    Args:
        model: Modelo PyTorch
        train_dataset: Dataset de treinamento
        val_dataset: Dataset de validação
        num_epochs: Número de épocas
    
    Returns:
        Histórico de métricas
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Usando dispositivo: {device}")
    
    # DataLoaders otimizados
    train_loader = create_optimized_dataloader(
        train_dataset, 
        batch_size=128,  # Pode aumentar se tiver memória
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = create_optimized_dataloader(
        val_dataset,
        batch_size=256,  # Batch maior para validação
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Configuração do otimizador e scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,  # Regularização
        betas=(0.9, 0.999)
    )
    
    scheduler = get_optimized_scheduler(
        optimizer, 
        scheduler_type='cosine',
        T_max=num_epochs,
        eta_min=1e-6
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    
    # Treinamento otimizado
    metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        # Parâmetros de otimização
        use_mixed_precision=True,
        gradient_accumulation_steps=1,  # Aumentar se precisar de batch maior
        max_grad_norm=1.0,
        early_stopping_patience=15,
        scheduler=scheduler,
        compile_model=True
    )
    
    return metrics
                    
                    
        
            
    