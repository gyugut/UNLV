from transformers import ConvNextForImageClassification as _ConvNextForImageClassification, ConvNextConfig
from ..base import BaseModel, ModelRegistry


class ConvNextForImageClassification(BaseModel, _ConvNextForImageClassification):
    model_id = "facebook/convnext-small-224"
    model_variation = ModelRegistry(**{
        'default': model_id, 'imagenet-1k': model_id, 'cifar-10': "clr4takeoff/convnext-small-cifar10"
    })


ConvNeXtSmall = ConvNextForImageClassification  # Alias


class ConvNextForImageClassification(BaseModel, _ConvNextForImageClassification):
    model_id = "facebook/convnext-tiny-224"
    model_variation = ModelRegistry(**{
        'default': model_id, 'imagenet-1k': model_id, 'cifar-10': "BeckerAnas/convnext-tiny-224-finetuned-cifar10"
    })


ConvNeXtTiny = ConvNextForImageClassification  # Alias