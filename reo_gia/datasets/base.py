from torch.utils.data import Dataset
from torchvision import transforms


class BaseDataset(Dataset):
    dataset_name: str = "dataset"
    num_classes: int = 0
    img_size: int = 0
    image_mean: list[float] = (0, 0, 0)
    image_std: list[float] = (1, 1, 1)

    @property
    def config(self) -> dict:
        return dict(
            size=self.img_size,
            do_resize=True,
            image_mean=self.image_mean,
            image_std=self.image_std
        )
