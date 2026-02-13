import os

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging


logger = logging.get_logger(__name__)


class ModelRegistry(dict):
    def __init__(self, default: str, **kwargs):
        super().__init__(**kwargs)
        self['default'] = default
        self.default = default
        self.__dict__.update(kwargs)


class BaseModel(PreTrainedModel):
    dataset_name = "default"
    model_id = os.path.join("local", "base")
    model_variation = ModelRegistry(default=model_id)

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        is_main_process: bool = True,
        state_dict: dict | None = None,
        push_to_hub: bool = False,
        max_shard_size: int | str = "50GB",
        variant: str | None = None,
        token: str | bool | None = None,
        save_peft_format: bool = True,
        save_original_format: bool = True,
        **kwargs,
    ):
        if not save_directory:
            save_directory = os.path.join(".", "results", f"{self.model_id}-{self.dataset_name}")

        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            save_original_format=save_original_format
            **kwargs
        )

        return save_directory

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        num_classes: int,
        *model_args,
        config: PretrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
        weights_only: bool = True,
        **kwargs,
    ) -> "BaseModel":
        if not pretrained_model_name_or_path:
            pretrained_model_name_or_path = cls.model_variation[cls.dataset_name]

        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs
        )
        model.model_id = pretrained_model_name_or_path

        if model.config.num_labels != num_classes:
            model.config.num_labels = num_classes
            model.classifier = cls(config=config).classifier
            model._init_weights(model.classifier)  # Reinitialize the classifier head
            logger.warning(f"Number of labels in the loaded model ({model.config.num_labels}) does not match the specified num_classes ({num_classes}). The classifier head has been reinitialized.")

        return model
