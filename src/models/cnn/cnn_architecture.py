import torch
import torch.nn as nn
from typing import List, Union
import random
from pydantic import BaseModel


class CNNBlock(nn.Module):
    """
    Bloco Convolucional Genérico.

    Consiste em uma sequência de Convolução 2D, Normalização em Lote (opcional),
    Função de Ativação e Dropout (opcional).
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
        dropout_rate: float = 0.0,
    ):
        """
        Inicializa o bloco convolucional.

        Args:
            in_channels: Número de canais de entrada.
            out_channels: Número de canais de saída.
            kernel_size: Tamanho do kernel da convolução (padrão: 3).
            stride: Stride da convolução (padrão: 1).
            padding: Padding da convolução (padrão: 1).
            activation_fn: Função de ativação (padrão: nn.ReLU).
            batch_norm: Se True, aplica normalização em lote.
            dropout_rate: Taxa de dropout (0-1, padrão: 0).
        """
        super().__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=not batch_norm,
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation_fn())
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.block = nn.Sequential(*layers)
        self._initialize_weights(activation_fn)

    def _initialize_weights(self, activation_fn: nn.Module) -> None:
        """
        Inicializa os pesos da convolução.

        Usa inicialização Kaiming para ativações ReLU-like (ReLU, LeakyReLU, ELU, SELU)
        e Xavier para outras. Bias é zerado se presente.

        Args:
            activation_fn: Função de ativação usada na camada.
        """
        for m in self.block:
            if isinstance(m, nn.Conv2d):
                if isinstance(activation_fn, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU)):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do bloco convolucional.

        Args:
            x: Tensor de entrada (formato: [batch, in_channels, altura, largura]).

        Returns:
            Tensor processado (formato: [batch, out_channels, altura', largura']).
        """
        return self.block(x)


class GenericCNN(nn.Module):
    """
    Implementação de uma Rede Neural Convolucional (CNN) genérica.

    Permite a construção de uma rede com múltiplos estágios convolucionais,
    seguidos por camadas densas para classificação.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        block_channels: List[int],
        kernel_sizes: Union[int, List[int]] = 3,
        strides: Union[int, List[int]] = 1,
        paddings: Union[int, List[int]] = 1,
        pooling_type: Union[str, None] = "max",
        pooling_kernel_size: int = 2,
        pooling_stride: int = 2,
        activation_fn: nn.Module = nn.ReLU,
        batch_norm: bool = True,
        dropout_rate: float = 0.0,
        fc_layers: List[int] = None,
    ):
        """
        Inicializa a CNN genérica.

        Args:
            in_channels: Número de canais de entrada (ex.: 3 para RGB).
            num_classes: Número de classes de saída.
            block_channels: Lista de canais de saída para cada estágio convolucional.
            kernel_sizes: Tamanho do kernel para cada camada convolucional. Pode ser um int
                          para todas as camadas ou uma lista de ints.
            strides: Stride para cada camada convolucional. Pode ser um int para todas
                     as camadas ou uma lista de ints.
            paddings: Padding para cada camada convolucional. Pode ser um int para todas
                      as camadas ou uma lista de ints.
            pooling_type: Tipo de pooling ('max', 'avg', ou None para não usar pooling).
            pooling_kernel_size: Tamanho do kernel para as camadas de pooling.
            pooling_stride: Stride para as camadas de pooling.
            activation_fn: Função de ativação (padrão: nn.ReLU).
            batch_norm: Se True, aplica normalização em lote.
            dropout_rate: Taxa de dropout (0-1, padrão: 0).
            fc_layers: Lista de tamanhos das camadas densas após as camadas convolucionais.
                       Se None ou vazia, nenhuma camada densa adicional é usada antes da camada final.

        Raises:
            ValueError: Se os parâmetros de listas (block_channels, kernel_sizes, etc.)
                        não forem consistentes ou inválidos.
        """
        super().__init__()

        if not isinstance(block_channels, list) or not all(
            isinstance(x, int) for x in block_channels
        ):
            raise ValueError("block_channels deve ser uma lista de inteiros.")
        if fc_layers is not None and (
            not isinstance(fc_layers, list)
            or not all(isinstance(x, int) for x in fc_layers)
        ):
            raise ValueError("fc_layers deve ser uma lista de inteiros ou None.")
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate deve estar entre 0 e 1.")

        num_conv_blocks = len(block_channels)

        # Trata kernel_sizes, strides e paddings para que sejam listas
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * num_conv_blocks
        if isinstance(strides, int):
            strides = [strides] * num_conv_blocks
        if isinstance(paddings, int):
            paddings = [paddings] * num_conv_blocks

        if not (
            len(kernel_sizes) == num_conv_blocks
            and len(strides) == num_conv_blocks
            and len(paddings) == num_conv_blocks
        ):
            raise ValueError(
                "As listas kernel_sizes, strides e paddings devem ter o mesmo comprimento que block_channels."
            )

        layers = []
        current_in_channels = in_channels

        for i, out_c in enumerate(block_channels):
            layers.append(
                CNNBlock(
                    current_in_channels,
                    out_c,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    activation_fn=activation_fn,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate,
                )
            )
            current_in_channels = out_c
            if pooling_type == "max":
                layers.append(
                    nn.MaxPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride)
                )
            elif pooling_type == "avg":
                layers.append(
                    nn.AvgPool2d(kernel_size=pooling_kernel_size, stride=pooling_stride)
                )

        self.features = nn.Sequential(*layers)

        # Camada de pooling global antes das camadas densas
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Camadas densas (Fully Connected)
        fc_layers_list = []
        # Para calcular o tamanho da entrada para a primeira camada FC
        # Precisamos de um forward pass temporário para determinar o número de features após as convoluções
        # Este é um truque comum para determinar o tamanho de entrada para as camadas FC dinamicamente
        with torch.no_grad():
            # Coloca o modelo em modo de avaliação para evitar problemas com BatchNorm
            self.features.eval()
            dummy_input = torch.randn(
                2, in_channels, 32, 32
            )  # Usando batch_size=2 para evitar problemas com BatchNorm
            dummy_output = self.features(dummy_input)
            dummy_output = self.avgpool(dummy_output)
            flattened_features = dummy_output.view(dummy_output.size(0), -1).shape[1]
            # Retorna ao modo de treinamento
            self.features.train()

        prev_fc_size = flattened_features
        if fc_layers:
            for fc_size in fc_layers:
                fc_layers_list.append(nn.Linear(prev_fc_size, fc_size))
                if batch_norm:  # Opcional: Batch norm em camadas densas
                    fc_layers_list.append(nn.BatchNorm1d(fc_size, track_running_stats=True))
                fc_layers_list.append(activation_fn())
                if dropout_rate > 0:
                    fc_layers_list.append(nn.Dropout(dropout_rate))
                prev_fc_size = fc_size

        self.classifier = nn.Sequential(*fc_layers_list)
        self.output_layer = nn.Linear(prev_fc_size, num_classes)
        self._initialize_fc_weights()

    def _initialize_fc_weights(self) -> None:
        """
        Inicializa os pesos das camadas Fully Connected.
        """
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da CNN genérica.

        Args:
            x: Tensor de entrada (formato: [batch, in_channels, altura, largura]).

        Returns:
            Tensor de logits (formato: [batch, num_classes]).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Achatando o tensor
        x = self.classifier(x)
        return self.output_layer(x)


class CNNParams(BaseModel):
    num_classes: int = 10
    min_channels: int = 32
    max_channels: int = 256
    dropout_rate: float = 0.0
    num_layers: int = 3
    batch_norm: bool = True


def generate_cnn_architecture(params: CNNParams) -> GenericCNN:
    """
    Gera uma instância de CNN com parâmetros simplificados.
    
    Args:
        params: Parâmetros da arquitetura contendo:
            - num_classes: Número de classes de saída
            - min_channels: Número mínimo de canais
            - max_channels: Número máximo de canais
            - dropout_rate: Taxa de dropout
            - num_layers: Número de camadas convolucionais
            - batch_norm: Se deve usar batch normalization
    """
    # Calcula o número de canais para cada camada
    channels = []
    step = (params.max_channels - params.min_channels) / (params.num_layers - 1)
    for i in range(params.num_layers):
        channels.append(int(params.min_channels + step * i))

    print(f"CNN: channels={channels}, dropout={params.dropout_rate}, num_layers={params.num_layers}")
    
    # Usa uma estratégia inteligente de pooling baseada no número de camadas
    # Para evitar que a resolução fique muito pequena
    if params.num_layers <= 4:
        # Poucas camadas: pode usar pooling normal
        pooling_type = "max"
    else:
        # Muitas camadas: reduz o pooling para evitar resolução muito pequena
        # Usa pooling apenas a cada 2-3 camadas ou sem pooling
        pooling_type = None  # Sem pooling, usa stride nas convoluções
    
    # Ajusta strides baseado no número de camadas
    if params.num_layers <= 4:
        strides = 1  # Stride normal com pooling
    else:
        # Para muitas camadas, usa stride 2 em algumas camadas para reduzir resolução
        strides = [2 if i in [1, 3, 5] else 1 for i in range(params.num_layers)]
    
    return GenericCNN(
        in_channels=3,
        num_classes=params.num_classes,
        block_channels=channels,
        kernel_sizes=3,
        strides=strides,
        paddings=1,
        pooling_type=pooling_type,
        pooling_kernel_size=2,
        pooling_stride=2,
        activation_fn=nn.ReLU,
        batch_norm=params.batch_norm,
        dropout_rate=params.dropout_rate,
        fc_layers=[128],
    )


if __name__ == "__main__":
    print("--- Testando a arquitetura CNN Genérica ---")

    # Teste 1: CNN básica com configuração padrão
    print("\n--- Teste 1: CNN Básica ---")
    try:
        params_padrao = CNNParams()
        model1 = generate_cnn_architecture(params_padrao)
        x1 = torch.randn(2, 3, 32, 32)  # batch_size=2
        output1 = model1(x1)
        print(f"Saída CNN Básica: {output1.shape}")
        assert output1.shape == torch.Size([2, 10]), "Erro no formato de saída da CNN Básica."
        print("CNN Básica testada com sucesso.")
    except Exception as e:
        print(f"Erro ao testar CNN Básica: {e}")

    # Teste 2: CNN com configuração personalizada
    print("\n--- Teste 2: CNN Complexa ---")
    try:
        params_custom = CNNParams(
            num_classes=100,
            min_channels=64,
            max_channels=512,
            dropout_rate=0.2,
            num_layers=5,
            batch_norm=True
        )
        model2 = generate_cnn_architecture(params_custom)
        x2 = torch.randn(2, 3, 64, 64)  # batch_size=2
        output2 = model2(x2)
        print(f"Saída CNN Complexa: {output2.shape}")
        assert output2.shape == torch.Size([2, 100]), "Erro no formato de saída da CNN Complexa."
        print("CNN Complexa testada com sucesso.")
    except Exception as e:
        print(f"Erro ao testar CNN Complexa: {e}")

    # Teste 3: CNN com configuração mínima
    print("\n--- Teste 3: CNN Mínima ---")
    try:
        params_min = CNNParams(
            num_classes=5,
            min_channels=8,
            max_channels=16,
            dropout_rate=0.0,
            num_layers=2,
            batch_norm=False
        )
        model3 = generate_cnn_architecture(params_min)
        x3 = torch.randn(2, 3, 28, 28)  # batch_size=2
        output3 = model3(x3)
        print(f"Saída CNN Mínima: {output3.shape}")
        assert output3.shape == torch.Size([2, 5]), "Erro no formato de saída da CNN Mínima."
        print("CNN Mínima testada com sucesso.")
    except Exception as e:
        print(f"Erro ao testar CNN Mínima: {e}")
