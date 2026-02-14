from transformers import ResNetForImageClassification as _ResNetForImageClassification, ResNetConfig
from ..base import BaseModel, ModelRegistry


class ResNetForImageClassification(BaseModel, _ResNetForImageClassification):
    model_id = "microsoft/resnet-34"
    model_variation = ModelRegistry(**{
        'default': model_id, 'imagenet-1k': model_id, 'cifar-10': "jialicheng/cifar10_resnet-34"
    })


ResNet34 = ResNetForImageClassification  # Alias


class ResNetForImageClassification(BaseModel, _ResNetForImageClassification):
    model_id = "microsoft/resnet-50"
    model_variation = ModelRegistry(**{
        'default': model_id, 'imagenet-1k': model_id, 'cifar-10': "jialicheng/cifar10_resnet-50"
    })


ResNet50 = ResNetForImageClassification  # Alias