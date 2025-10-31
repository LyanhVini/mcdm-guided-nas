import torch
import torch.nn as nn
from typing import List
import random
from pydantic import BaseModel


class ResNetInputLayer(nn.Module):
    """
    Camada inicial da ResNet.

    Aplica uma convolução inicial, normalização em lote (opcional) e função de ativação
    para processar a entrada da rede (ex.: imagens RGB do CIFAR-10).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        activation_fn: nn.Module = nn.ReLU,
        batch_norm: bool = True,
    ):
        """
        Inicializa a camada inicial.

        Args:
            in_channels: Número de canais de entrada (ex.: 3 para RGB).
            out_channels: Número de canais de saída.
            kernel_size: Tamanho do kernel da convolução (padrão: 3).
            stride: Stride da convolução (padrão: 1).
            padding: Padding da convolução (padrão: 1).
            activation_fn: Função de ativação (padrão: nn.ReLU).
            batch_norm: Se True, aplica normalização em lote.
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm
        )
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation_fn()
        self._initialize_weights(activation_fn)

    def _initialize_weights(self, activation_fn: nn.Module) -> None:
        """
        Inicializa os pesos da convolução.

        Usa inicialização Kaiming para ativações ReLU-like (ReLU, LeakyReLU, ELU, SELU)
        e Xavier para outras. Bias é zerado se presente.

        Args:
            activation_fn: Função de ativação usada na camada.
        """
        if isinstance(activation_fn, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU)):
            nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        else:
            nn.init.xavier_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da camada inicial.

        Aplica convolução, normalização em lote (se habilitada) e ativação.

        Args:
            x: Tensor de entrada (formato: [batch, in_channels, altura, largura]).

        Returns:
            Tensor processado (formato: [batch, out_channels, altura', largura']).
        """
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        return self.activation(x)


class BasicBlock(nn.Module):
    """
    Bloco residual básico para ResNet-18/34.

    Contém duas convoluções 3x3 com conexão residual, suportando dropout,
    normalização em lote e ativação customizada.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation_fn: nn.Module = nn.ReLU,
        dropout_rate: float = 0.0,
        batch_norm: bool = True,
    ):
        """
        Inicializa o bloco residual básico.

        Args:
            in_channels: Número de canais de entrada.
            out_channels: Número de canais de saída.
            stride: Stride da primeira convolução (padrão: 1).
            activation_fn: Função de ativação (padrão: nn.ReLU).
            dropout_rate: Taxa de dropout (0-1, padrão: 0).
            batch_norm: Se True, aplica normalização em lote.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not batch_norm,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation_fn()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not batch_norm,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=not batch_norm,
                ),
                (
                    nn.BatchNorm2d(out_channels * self.expansion)
                    if batch_norm
                    else nn.Identity()
                ),
            )
        self._initialize_weights(activation_fn)

    def _initialize_weights(self, activation_fn: nn.Module) -> None:
        """
        Inicializa os pesos das convoluções.

        Usa Kaiming para ReLU-like e Xavier para outras ativações. Bias é zerado.

        Args:
            activation_fn: Função de ativação usada no bloco.
        """
        for conv in [self.conv1, self.conv2]:
            if isinstance(activation_fn, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU)):
                nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            else:
                nn.init.xavier_normal_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do bloco residual.

        Aplica conv1 -> bn1 -> ativação -> dropout -> conv2 -> bn2, soma com
        o caminho residual (shortcut) e aplica ativação final.

        Args:
            x: Tensor de entrada (formato: [batch, in_channels, altura, largura]).

        Returns:
            Tensor processado (formato: [batch, out_channels, altura', largura']).
        """
        identity = self.shortcut(x)
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x += identity
        return self.activation(x)


class Bottleneck(nn.Module):
    """
    Bloco Bottleneck para ResNet-50/101/152.

    Contém três convoluções (1x1, 3x3, 1x1) com conexão residual, suportando
    dropout, normalização em lote e ativação customizada.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation_fn: nn.Module = nn.ReLU,
        dropout_rate: float = 0.0,
        batch_norm: bool = True,
    ):
        """
        Inicializa o bloco Bottleneck.

        Args:
            in_channels: Número de canais de entrada.
            out_channels: Número de canais de saída (antes da expansão).
            stride: Stride da convolução 3x3 (padrão: 1).
            activation_fn: Função de ativação (padrão: nn.ReLU).
            dropout_rate: Taxa de dropout (0-1, padrão: 0).
            batch_norm: Se True, aplica normalização em lote.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=not batch_norm
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.activation = activation_fn()
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=not batch_norm,
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=not batch_norm,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion) if batch_norm else None
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=not batch_norm,
                ),
                (
                    nn.BatchNorm2d(out_channels * self.expansion)
                    if batch_norm
                    else nn.Identity()
                ),
            )
        self._initialize_weights(activation_fn)

    def _initialize_weights(self, activation_fn: nn.Module) -> None:
        """
        Inicializa os pesos das convoluções.

        Usa Kaiming para ReLU-like e Xavier para outras ativações. Bias é zerado.

        Args:
            activation_fn: Função de ativação usada no bloco.
        """
        for conv in [self.conv1, self.conv2, self.conv3]:
            if isinstance(activation_fn, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU)):
                nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            else:
                nn.init.xavier_normal_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do bloco Bottleneck.

        Aplica conv1 -> bn1 -> act -> dropout -> conv2 -> bn2 -> act -> dropout ->
        conv3 -> bn3, soma com o caminho residual e aplica ativação final.

        Args:
            x: Tensor de entrada (formato: [batch, in_channels, altura, largura]).

        Returns:
            Tensor processado (formato: [batch, out_channels * expansion, altura', largura']).
        """
        identity = self.shortcut(x)
        x = self.conv1(x)
        if self.bn1 is not None:
            x = self.bn1(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv2(x)
        if self.bn2 is not None:
            x = self.bn2(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv3(x)
        if self.bn3 is not None:
            x = self.bn3(x)
        x += identity
        return self.activation(x)


class ResNetOutputLayer(nn.Module):
    """
    Camada de saída da ResNet.

    Aplica average pooling global e uma camada linear para mapear as features
    para o espaço de classes.
    """

    def __init__(self, in_channels: int, num_classes: int):
        """
        Inicializa a camada de saída.

        Args:
            in_channels: Número de canais de entrada (saída do último estágio).
            num_classes: Número de classes de saída.
        """
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(in_channels, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Inicializa os pesos da camada linear.

        Usa inicialização Xavier para os pesos e zera o bias.
        """
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da camada de saída.

        Aplica average pooling, achata o tensor e passa pela camada linear.

        Args:
            x: Tensor de entrada (formato: [batch, in_channels, altura, largura]).

        Returns:
            Tensor de logits (formato: [batch, num_classes]).
        """
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


class ResNet(nn.Module):
    """
    Implementação da Residual Network (ResNet).

    Consiste em uma camada inicial, quatro estágios residuais (com blocos BasicBlock
    ou Bottleneck) e uma camada de saída. Suporta configurações flexíveis.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        block_type: nn.Module,
        layers: List[int],
        num_classes: int,
        activation_fn: nn.Module = nn.ReLU,
        dropout_rate: float = 0.0,
        batch_norm: bool = True,
    ):
        """
        Inicializa a ResNet.

        Args:
            in_channels: Número de canais de entrada (ex.: 3 para RGB).
            base_channels: Número de canais base da camada inicial.
            block_type: Tipo de bloco residual (BasicBlock ou Bottleneck).
            layers: Lista com número de blocos por estágio (ex.: [2, 2, 2, 2]).
            num_classes: Número de classes de saída.
            activation_fn: Função de ativação (padrão: nn.ReLU).
            dropout_rate: Taxa de dropout (0-1, padrão: 0).
            batch_norm: Se True, aplica normalização em lote.

        Raises:
            ValueError: Se layers não for lista de inteiros, dropout_rate inválido,
                        ou block_type inválido.
        """
        super().__init__()
        if not isinstance(layers, list) or not all(isinstance(x, int) for x in layers):
            raise ValueError("layers deve ser uma lista de inteiros")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate deve estar entre 0 e 1")
        if block_type not in [BasicBlock, Bottleneck]:
            raise ValueError("block_type deve ser BasicBlock ou Bottleneck")

        self.in_channels = base_channels
        self.input_layer = ResNetInputLayer(
            in_channels, base_channels, 3, 1, 1, activation_fn, batch_norm
        )
        self.stage1 = self._make_stage(
            block_type,
            base_channels,
            layers[0],
            1,
            activation_fn,
            dropout_rate,
            batch_norm,
        )
        self.stage2 = self._make_stage(
            block_type,
            base_channels * 2,
            layers[1],
            2,
            activation_fn,
            dropout_rate,
            batch_norm,
        )
        self.stage3 = self._make_stage(
            block_type,
            base_channels * 4,
            layers[2],
            2,
            activation_fn,
            dropout_rate,
            batch_norm,
        )
        self.stage4 = self._make_stage(
            block_type,
            base_channels * 8,
            layers[3],
            2,
            activation_fn,
            dropout_rate,
            batch_norm,
        )
        self.output_layer = ResNetOutputLayer(
            base_channels * 8 * block_type.expansion, num_classes
        )

    def _make_stage(
        self,
        block_type: nn.Module,
        out_channels: int,
        num_blocks: int,
        stride: int,
        activation_fn: nn.Module,
        dropout_rate: float,
        batch_norm: bool,
    ) -> nn.Sequential:
        """
        Cria um estágio da ResNet com blocos residuais.

        Args:
            block_type: Tipo de bloco (BasicBlock ou Bottleneck).
            out_channels: Número de canais de saída (antes da expansão).
            num_blocks: Número de blocos no estágio.
            stride: Stride do primeiro bloco.
            activation_fn: Função de ativação.
            dropout_rate: Taxa de dropout.
            batch_norm: Se True, aplica normalização em lote.

        Returns:
            nn.Sequential contendo os blocos residuais.
        """
        layers = []
        layers.append(
            block_type(
                self.in_channels,
                out_channels,
                stride,
                activation_fn,
                dropout_rate,
                batch_norm,
            )
        )
        self.in_channels = out_channels * block_type.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block_type(
                    self.in_channels,
                    out_channels,
                    1,
                    activation_fn,
                    dropout_rate,
                    batch_norm,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da ResNet.

        Processa a entrada através da camada inicial, estágios residuais e camada
        de saída, retornando logits.

        Args:
            x: Tensor de entrada (formato: [batch, in_channels, altura, largura]).

        Returns:
            Tensor de logits (formato: [batch, num_classes]).
        """
        x = self.input_layer(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.output_layer(x)


class ResNetParams(BaseModel):
    num_classes: int = 10
    min_channels: int = 16
    max_channels: int = 128
    dropout_rate: float = 0.0
    num_layers: int = 4
    batch_norm: bool = True

def generate_resnet_architecture(params: ResNetParams) -> ResNet:
    """
    Gera uma instância ResNet com parâmetros simplificados seguindo o padrão MobileNet/CNN.
    
    Args:
        params: Parâmetros da arquitetura contendo:
            - num_classes: Número de classes de saída
            - min_channels: Número mínimo de canais base
            - max_channels: Número máximo de canais base
            - dropout_rate: Taxa de dropout
            - num_layers: Número de estágios (camadas)
            - batch_norm: Se deve usar batch normalization
    """
    # Ajusta parâmetros específicos para ResNet (evita problemas de memória)
    adjusted_params = adjust_resnet_params_for_memory(params)
    
    # Calcula o número de canais base
    base_channels = adjusted_params.min_channels + (adjusted_params.max_channels - adjusted_params.min_channels) // 2
    
    # Configura o número de blocos por estágio baseado no número de camadas
    # Garante que sempre tenha pelo menos 4 estágios para ResNet
    if adjusted_params.num_layers < 4:
        layers = [2, 2, 2, 2]  # ResNet-32 padrão
    elif adjusted_params.num_layers == 4:
        layers = [2, 2, 2, 2]  # ResNet-32
    elif adjusted_params.num_layers == 5:
        layers = [2, 2, 2, 2, 2]  # ResNet-44
    else:
        # Para mais de 5 camadas, distribui os blocos nos 4 estágios
        total_blocks = min(adjusted_params.num_layers, 6)  # Limita a 6 blocos para ResNet
        layers = [max(1, total_blocks // 4)] * 4
        # Distribui os blocos restantes
        remaining = total_blocks % 4
        for i in range(remaining):
            layers[i] += 1
    
    print(f"ResNet: base_channels={base_channels}, layers={layers}, dropout={adjusted_params.dropout_rate}, num_layers={adjusted_params.num_layers}")
    
    return ResNet(
        in_channels=3,
        base_channels=base_channels,
        block_type=BasicBlock,  # Usa BasicBlock por padrão para ser mais leve
        layers=layers,
        num_classes=adjusted_params.num_classes,
        activation_fn=nn.ReLU,
        dropout_rate=adjusted_params.dropout_rate,
        batch_norm=adjusted_params.batch_norm,
    )

def adjust_resnet_params_for_memory(params: ResNetParams) -> ResNetParams:
    """
    Ajusta os parâmetros da ResNet para evitar problemas de memória GPU.
    Aplica restrições específicas para ResNet sem afetar outras arquiteturas.
    
    Args:
        params: Parâmetros originais da ResNet
        
    Returns:
        ResNetParams: Parâmetros ajustados para ResNet
    """
    adjusted_params = params.copy()
    
    # Limita max_channels para ResNet (mais conservador que MobileNet)
    if adjusted_params.max_channels > 512:
        print(f"⚠️  ResNet: limitando max_channels de {adjusted_params.max_channels} para 512")
        adjusted_params.max_channels = 512
    
    # Limita min_channels para ResNet
    if adjusted_params.min_channels > 64:
        print(f"⚠️  ResNet: limitando min_channels de {adjusted_params.min_channels} para 64")
        adjusted_params.min_channels = 64
    
    # Limita num_layers para ResNet (mais conservador)
    if adjusted_params.num_layers > 12:
        print(f"⚠️  ResNet: limitando num_layers de {adjusted_params.num_layers} para 12")
        adjusted_params.num_layers = 12
    
    return adjusted_params

