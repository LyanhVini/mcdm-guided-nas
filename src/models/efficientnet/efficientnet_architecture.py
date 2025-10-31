import math
import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Union, Type
from pydantic import BaseModel

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class HardSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.nn.functional.hardtanh(x + 3, 0., 6.) / 6.

class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.nn.functional.softplus(x))

ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'swish': Swish,
    'hardswish': HardSwish,
    'mish': Mish,
    'silu': nn.SiLU,
    'leakyrelu': lambda: nn.LeakyReLU(negative_slope=0.1)
}

class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        squeeze_factor: int = 4,
        activation_fn: nn.Module = Swish(),
        gate_fn: nn.Module = nn.Sigmoid()
    ):
        super().__init__()
        squeeze_channels = max(1, in_channels // squeeze_factor)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeeze_channels, 1),
            activation_fn,
            nn.Conv2d(squeeze_channels, in_channels, 1),
            gate_fn
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)

class MBConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expansion_factor: int,
        se_ratio: float = 0.25,
        dropout_rate: float = 0.2,
        activation_fn: nn.Module = Swish(),
        use_batch_norm: bool = True,
        batch_norm_momentum: float = 0.1,
        batch_norm_epsilon: float = 1e-5,
        se_gate_fn: nn.Module = nn.Sigmoid()
    ):
        super().__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        self.dropout_rate = dropout_rate
        expanded_channels = in_channels * expansion_factor

        norm_layer = lambda c: nn.BatchNorm2d(c, momentum=batch_norm_momentum, eps=batch_norm_epsilon) if use_batch_norm else nn.Identity()

        layers = []
        
        # Expansion phase
        if expansion_factor != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                norm_layer(expanded_channels),
                activation_fn
            ])

        # Depthwise conv
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size,
                     stride=stride, padding=kernel_size//2, groups=expanded_channels,
                     bias=False),
            norm_layer(expanded_channels),
            activation_fn
        ])

        # Squeeze and Excitation
        if se_ratio > 0:
            layers.append(SqueezeExcitation(
                expanded_channels,
                squeeze_factor=int(1/se_ratio),
                activation_fn=activation_fn,
                gate_fn=se_gate_fn
            ))

        # Projection phase
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            if self.training and self.dropout_rate > 0:
                keep_prob = 1 - self.dropout_rate
                mask = torch.rand([x.shape[0], 1, 1, 1], device=x.device) < keep_prob
                result = self.layers(x)
                scaled_result = result / keep_prob
                output = torch.where(mask, scaled_result, torch.zeros_like(scaled_result))
                return x + output
            else:
                return x + self.layers(x)
        else:
            return self.layers(x)

class EfficientNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        layers: List[int] = [1, 2, 2, 3, 3, 4, 1],
        num_classes: int = 10,
        activation_fn: nn.Module = nn.SiLU(),
        dropout_rate: float = 0.2,
        batch_norm: bool = True,
        se_ratio: float = 0.25,
        drop_connect_rate: float = 0.2,
    ):
        super().__init__()
        
        # Configuração base dos estágios EfficientNet
        # (kernel_size, expansion_factor, stride)
        stage_configs = [
            (3, 1, 1),    # Stage 1: stem
            (3, 6, 1),    # Stage 2
            (5, 6, 2),    # Stage 3
            (3, 6, 2),    # Stage 4
            (5, 6, 2),    # Stage 5
            (5, 6, 1),    # Stage 6
            (3, 6, 1),    # Stage 7
        ]
        
        # Limita o número de estágios baseado no número de camadas
        num_stages = min(len(stage_configs), len(layers))
        stage_configs = stage_configs[:num_stages]
        layers = layers[:num_stages]
        
        # Camada inicial (stem)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels) if batch_norm else nn.Identity(),
            activation_fn
        )
        
        # Blocos MBConv
        blocks = []
        in_channels = base_channels
        total_blocks = sum(layers)
        block_idx = 0
        
        for stage_idx, (kernel_size, expansion_factor, stride) in enumerate(stage_configs):
            out_channels = base_channels * (2 ** stage_idx)
            
            for layer_idx in range(layers[stage_idx]):
                # Calcula drop connect rate
                drop_rate = drop_connect_rate * block_idx / total_blocks
                
                # Usa stride apenas na primeira camada de cada estágio
                curr_stride = stride if layer_idx == 0 else 1
                
                blocks.append(
                    MBConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=curr_stride,
                        expansion_factor=expansion_factor,
                        se_ratio=se_ratio,
                        dropout_rate=drop_rate,
                        activation_fn=activation_fn,
                        use_batch_norm=batch_norm,
                        batch_norm_momentum=0.1,
                        batch_norm_epsilon=1e-5,
                    )
                )
                in_channels = out_channels
                block_idx += 1
        
        self.blocks = nn.Sequential(*blocks)
        
        # Camada final
        final_channels = in_channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels) if batch_norm else nn.Identity(),
            activation_fn
        )
        
        # Classificador
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(final_channels, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa os pesos do modelo."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.classifier(x)
        return x

def adjust_efficientnet_params_for_memory(params):
    """
    Ajusta os parâmetros da EfficientNet para evitar problemas de memória GPU.
    Aplica restrições específicas para EfficientNet sem afetar outras arquiteturas.
    
    Args:
        params: Parâmetros originais da EfficientNet
        
    Returns:
        EfficientNetParams: Parâmetros ajustados para EfficientNet
    """
    adjusted_params = params.copy()
    
    # Limita max_channels para EfficientNet (mais conservador que MobileNet)
    if adjusted_params.max_channels > 1024:
        print(f"⚠️  EfficientNet: limitando max_channels de {adjusted_params.max_channels} para 1024")
        adjusted_params.max_channels = 1024
    
    # Limita min_channels para EfficientNet
    if adjusted_params.min_channels > 32:
        print(f"⚠️  EfficientNet: limitando min_channels de {adjusted_params.min_channels} para 32")
        adjusted_params.min_channels = 32
    
    # Limita num_layers para EfficientNet (mais conservador)
    if adjusted_params.num_layers > 7:
        print(f"⚠️  EfficientNet: limitando num_layers de {adjusted_params.num_layers} para 7")
        adjusted_params.num_layers = 7
    
    return adjusted_params

class EfficientNetParams(BaseModel):
    num_classes: int = 10
    min_channels: int = 32
    max_channels: int = 128
    dropout_rate: float = 0.2
    num_layers: int = 4
    batch_norm: bool = True

def generate_efficientnet_architecture(params: EfficientNetParams) -> EfficientNet:
    """
    Gera uma instância EfficientNet com parâmetros simplificados seguindo o padrão MobileNet/CNN.
    
    Args:
        params: Parâmetros da arquitetura contendo:
            - num_classes: Número de classes de saída
            - min_channels: Número mínimo de canais base
            - max_channels: Número máximo de canais base
            - dropout_rate: Taxa de dropout
            - num_layers: Número de blocos MBConv
            - batch_norm: Se deve usar batch normalization
    """
    # Ajusta parâmetros específicos para EfficientNet (evita problemas de memória)
    adjusted_params = adjust_efficientnet_params_for_memory(params)
    
    # Calcula o número de canais base
    base_channels = adjusted_params.min_channels + (adjusted_params.max_channels - adjusted_params.min_channels) // 2
    
    # Configura a arquitetura EfficientNet baseada no número de camadas
    if adjusted_params.num_layers < 3:
        layers = [1, 1, 1]  # EfficientNet mínima
    elif adjusted_params.num_layers == 3:
        layers = [1, 1, 1]  # EfficientNet-3
    elif adjusted_params.num_layers == 5:
        layers = [1, 2, 2]  # EfficientNet-5
    elif adjusted_params.num_layers == 7:
        layers = [2, 2, 3]  # EfficientNet-7
    else:
        # Para mais de 7 camadas, distribui os blocos
        total_blocks = min(adjusted_params.num_layers, 9)  # Limita a 9 blocos para EfficientNet
        layers = [max(1, total_blocks // 3)] * 3
        # Distribui os blocos restantes
        remaining = total_blocks % 3
        for i in range(remaining):
            layers[i] += 1
    
    print(f"EfficientNet: base_channels={base_channels}, layers={layers}, dropout={adjusted_params.dropout_rate}, num_layers={adjusted_params.num_layers}")
    
    return EfficientNet(
        in_channels=3,
        base_channels=base_channels,
        layers=layers,
        num_classes=adjusted_params.num_classes,
        activation_fn=nn.SiLU(),  # EfficientNet usa SiLU/Swish (instância)
        dropout_rate=adjusted_params.dropout_rate,
        batch_norm=adjusted_params.batch_norm,
    )

if __name__ == "__main__":
    # Example usage with new standardized parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a customized EfficientNet model
    params = EfficientNetParams(
        num_classes=10,
        min_channels=32,
        max_channels=128,
        dropout_rate=0.2,
        num_layers=4,
        batch_norm=True
    )
    
    model = generate_efficientnet_architecture(params).to(device)
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 32, 32).to(device)  # CIFAR-10 size
    output = model(input_tensor)
    
    print(f"Model Output Shape: {output.shape}")  # Should be [4, 10]
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")
