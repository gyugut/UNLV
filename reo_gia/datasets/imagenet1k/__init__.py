from typing import Callable, Optional
from os import path, rmdir, mkdir
from pathlib import Path
import traceback

from torchvision.datasets.utils import extract_archive
from torchvision import datasets
import huggingface_hub

from .classes import IMAGENET2012_CLASSES


class ImageNet1K(datasets.ImageFolder):
    """
    ImageNet-1K dataset

    :ref: https://huggingface.co/datasets/ILSVRC/imagenet-1k
    """

    download_method = huggingface_hub.hf_hub_download
    dataset_name = "imagenet-1k"
    dataset_id = "ILSVRC/imagenet-1k"
    revision = "1500f8c59b214ce459c0a593fa1c87993aeb7700"
    obj_classes = tuple(IMAGENET2012_CLASSES.keys())
    obj_class_namees = tuple(IMAGENET2012_CLASSES.values())
    num_classes = len(obj_classes)

    img_size = 224
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)

    filenames = {
        "train": [f"train_images_{i}.tar.gz" for i in range(5)],
        "val": ["val_images.tar.gz"],
        "test": ["test_images.tar.gz"],
    }

    def __init__(
        self,
        root: str,
        force_download: bool = True,
        train: bool = True,
        valid: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        self.root = path.join(root, self.dataset_name)

        if train:
            self.root = path.join(self.root, "val") if valid else path.join(self.root, "train")
        else:
            self.root = path.join(self.root, "test")

        self._download(self.root, force=force_download)
        super().__init__(root=self.root, transform=transform, target_transform=target_transform)

    @classmethod
    def _download(cls, root: str, force: bool = False):
        root = Path(root)

        # Do download if the dataset does not exist
        print(f"INFO: Downloading '{cls.dataset_id}' dataset from huggingface to {root}...")
        dataset_type = "train" if "train" in str(root) else "val" if "val" in str(root) else "test"

        if force or not (
                path.exists(root) and any(p for p in root.iterdir() if not p.name.startswith('.'))
        ):  # Check if dataset files already exist in the directory
            [cls.download_method(
                filename=filename,
                subfolder="data",
                repo_id=cls.dataset_id,
                repo_type="dataset",
                local_dir=root.parent,
                force_download=force,
                revision=cls.revision
            ) for filename in cls.filenames[dataset_type]]

            for filename in cls.filenames[dataset_type]:
                from_path = path.join(root.parent, "data", filename)
                extract_archive(from_path, root, remove_finished=True)

            if path.exists(path.join(root, "data")):
                rmdir(path.join(root, "data"))

            for filename in list(root.iterdir()):
                if filename.name.endswith(".JPEG") or filename.name.endswith(".jpeg"):
                    if dataset_type == "train":
                        class_name = filename.name.split("_")[0]
                        if not path.isdir(path.join(root, class_name)):
                            mkdir(path.join(root, class_name))
                        new_name = filename.name.split("_")[1] + ".JPEG"
                        filename.rename(path.join(root, class_name, new_name))
                    elif dataset_type == "val":
                        class_name = filename.name.replace(".JPEG", "").replace(".jpeg", "").split("_")[-1]
                        if not path.isdir(path.join(root, class_name)):
                            mkdir(path.join(root, class_name))
                        new_name = filename.name.replace("_"+class_name, "")
                        filename.rename(path.join(root, class_name, new_name))
                    else:
                        if not path.isdir(path.join(root, "images")):
                            mkdir(path.join(root, "images"))
                        filename.rename(path.join(filename.parent, "images", filename.name))
            print("INFO: Dataset downloaded successfully.")
        else:
            print("INFO: Dataset files found in the root directory. Skipping download.")
