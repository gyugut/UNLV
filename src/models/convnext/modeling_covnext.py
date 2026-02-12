from transformers import ConvNextConfig, ConvNextModel

from ..base import BaseModel
import torch.nn as nn


class ConvNeXt(BaseModel):
    def __init__(self, config: ConvNextConfig, num_classes: int):
        super().__init__(image_size=config.image_size, num_classes=num_classes)
        self.config = config
        self.model = ConvNextModel(config=config)
        self.fc = nn.Linear(self.config.hidden_sizes[-1], num_classes)

    def forward(self, x, *args, **kwargs):
        out = self.model(x)
        pooled = out.pooler_output  # [batch_size, hidden_size]
        logits = self.fc(pooled)  # [batch_size, num_classes]
        return logits


class ConvNeXtTiny(ConvNeXt):
    model_name = "ConvNeXt-Tiny"

    def __init__(self, image_size: int, num_classes: int):
        config = ConvNextConfig.from_pretrained("facebook/convnext-tiny-224")
        super().__init__(config=config, num_classes=num_classes)


class ConvNeXtSmall(ConvNeXt):
    model_name = "ConvNeXt-Small"

    def __init__(self, image_size: int, num_classes: int):
        config = ConvNextConfig.from_pretrained("facebook/convnext-small-224")
        super().__init__(config=config, num_classes=num_classes)


class ConvNeXtBase(ConvNeXt):
    model_name = "ConvNeXt-Base"

    def __init__(self, image_size: int, num_classes: int):
        config = ConvNextConfig.from_pretrained("facebook/convnext-base-224")
        super().__init__(config=config, num_classes=num_classes)


class ConvNeXtLarge(ConvNeXt):
    model_name = "ConvNeXt-Large"

    def __init__(self, image_size: int, num_classes: int):
        config = ConvNextConfig.from_pretrained("facebook/convnext-large-224")
        super().__init__(config=config, num_classes=num_classes)