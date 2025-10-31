"""
Carregador de Arquiteturas
=========================
"""
from models.mobilenet.mobilenet_architecture import generate_mobilenet_architecture, MobilenetParams
from models.mobilenet.mobilenet_warm_up import warm_up_mobilenet
from models.cnn.cnn_architecture import generate_cnn_architecture, CNNParams
from models.cnn.cnn_warm_up import warm_up_cnn
from models.resnet.resnet_architecture import generate_resnet_architecture, ResNetParams
from models.resnet.resnet_warm_up import warm_up_resnet
from models.efficientnet.efficientnet_architecture import generate_efficientnet_architecture, EfficientNetParams
from models.efficientnet.efficientnet_warm_up import warm_up_efficientnet

archictectures = {
    "MobileNet": {
        "params": MobilenetParams(),
        "generate_architecture": generate_mobilenet_architecture,
        "warm_up": warm_up_mobilenet
    },
    "CNN": {
        "params": CNNParams(),
        "generate_architecture": generate_cnn_architecture,
        "warm_up": warm_up_cnn
    },
    "ResNet": {
        "params": ResNetParams(),
        "generate_architecture": generate_resnet_architecture,
        "warm_up": warm_up_resnet
    },
    "EfficientNet": {
        "params": EfficientNetParams(),
        "generate_architecture": generate_efficientnet_architecture,
        "warm_up": warm_up_efficientnet
    }
}