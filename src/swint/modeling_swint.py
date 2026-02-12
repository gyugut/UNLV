from transformers import SwinConfig, SwinModel
from ..base import BaseModel
import torch.nn as nn


class SwinTransformer(BaseModel, ):
    PRETRAINED_MODEL_NAME = None

    def __init__(self, config: SwinConfig, num_classes: int):
        super().__init__(image_size=config.image_size, num_classes=num_classes)
        self.config = config
        self.model = SwinModel.from_pretrained(self.PRETRAINED_MODEL_NAME)
        self.fc = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, x, *args, **kwargs):
        out = self.model(x)
        pooled = out.pooler_output  # [batch_size, hidden_size]
        logits = self.fc(pooled)  # [batch_size, num_classes]
        return logits


class SwinTiny(SwinTransformer):
    model_name = "Swin-Tiny_P4W7"
    PRETRAINED_MODEL_NAME = "microsoft/swin-tiny-patch4-window7-224"

    def __init__(self, image_size: int, num_classes: int):
        config = SwinConfig.from_pretrained(self.PRETRAINED_MODEL_NAME)
        super().__init__(config=config, num_classes=num_classes)


class SwinSmall(SwinTransformer):
    model_name = "Swin-Small_P4W7"
    PRETRAINED_MODEL_NAME = "microsoft/swin-small-patch4-window7-224"

    def __init__(self, image_size: int, num_classes: int):
        config = SwinConfig.from_pretrained(self.PRETRAINED_MODEL_NAME)
        super().__init__(config=config, num_classes=num_classes)


class SwinBase(SwinTransformer):
    model_name = "Swin-Base_P4W7"
    PRETRAINED_MODEL_NAME = "microsoft/swin-base-patch4-window7-224"

    def __init__(self, image_size: int, num_classes: int):
        config = SwinConfig.from_pretrained(self.PRETRAINED_MODEL_NAME)
        super().__init__(config=config, num_classes=num_classes)


class SwinLarge(SwinTransformer):
    model_name = "Swin-Large_P4W7"
    PRETRAINED_MODEL_NAME = "microsoft/swin-large-patch4-window7-224"

    def __init__(self, image_size: int, num_classes: int):
        config = SwinConfig.from_pretrained(self.PRETRAINED_MODEL_NAME)
        super().__init__(config=config, num_classes=num_classes)