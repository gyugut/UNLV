from transformers import SwinForImageClassification as _SwinForImageClassification, SwinConfig
from ..base import BaseModel, ModelRegistry


class SwinForImageClassification(BaseModel, _SwinForImageClassification):
    model_id = "microsoft/swin-tiny-patch4-window7-224"
    model_variation = ModelRegistry(**{
        'default': model_id, 'imagenet-1k': model_id, 'cifar-10': ""
    })


SwinTiny = SwinForImageClassification  # Alias


class SwinForImageClassification(BaseModel, _SwinForImageClassification):
    model_id = "microsoft/swin-small-patch4-window7-224"
    model_variation = ModelRegistry(**{
        'default': model_id, 'imagenet-1k': model_id, 'cifar-10': ""
    })


SwinSmall = SwinForImageClassification  # Alias
