import logging
from pathlib import Path
from typing import Iterable, Union

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

PathLike = Union[Path, str]
input_shape = 1, 3, 240, 320
logger = logging.getLogger(__name__)

cv2.setNumThreads(0)


class ImagePoseDataset(Dataset):
    def __init__(self, image_dir: PathLike, transform=None, use_cache: bool = True):
        from glob import glob

        image_dir = Path(image_dir)
        csv_file = list(image_dir.glob("*.csv"))[0]
        self.annotations = pd.read_csv(csv_file)

        self.transform = transform
        # Using glob instead of Path.rglob() that doesn't follow symlink
        self.filenames = [
            Path(s) for s in glob(f"{image_dir}/**/*.jpg", recursive=True)
        ]
        self.use_cache = use_cache
        self.cache = [None for _ in self.filenames]
        logger.info(f"Located {len(self.filenames)} files in {image_dir}")

    def _get_label(self, img_path: Path):
        import numpy as np

        img_name = img_path.name
        annotation = self.annotations.loc[
            self.annotations["image_name"] == img_name
        ].iloc[0]
        heading = np.rad2deg(annotation["heading"])
        dl = annotation["distance_left"]
        dr = annotation["distance_right"]
        distance = dl / (dl + dr)
        return heading, distance

    @staticmethod
    def _get_image(img_path: Path):
        image = cv2.cvtColor(cv2.imread(img_path.as_posix()), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        assert h == 720 and w == 1280
        # Crop to 960 * 720
        target_w = 960
        margin = (w - target_w) // 2
        image = image[:, margin:-margin]
        return cv2.resize(image, (input_shape[3], input_shape[2]))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = self.filenames[idx]
        if not self.use_cache:
            image, target = self._get_image(path), self._get_label(path)
        elif self.cache[idx] is None:
            image, target = self._get_image(path), self._get_label(path)
            self.cache[idx] = image, target
        else:
            image, target = self.cache[idx]

        if self.transform:
            image, target = self.transform(image, target)
        return image, target

    def distill_to_pickle(self, output_file: PathLike, indices: Iterable[int]):
        import pickle as pkl

        from tqdm import tqdm

        inputs, labels = zip(*[self[idx] for idx in tqdm(indices)])
        with open(output_file, "wb") as f:
            pkl.dump({"inputs": inputs, "labels": labels}, f)


def get_dataset_tr(augment: bool):
    from torchvision.transforms import transforms

    basic_tr = TargetTrCompose(
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )
    if not augment:
        return basic_tr
    return TargetTrCompose(
        transforms.ToPILImage(),
        transforms.RandomApply([transforms.ColorJitter(0.5, 0.25, 0.25, 0.1)], 0.5),
        RandomHorizontalFlip(),
        basic_tr,
    )


class TargetTrCompose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            try:
                img, target = t(img, target)
            except TypeError:
                img = t(img)
        return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, target):
        import torchvision.transforms.functional as F

        if torch.rand(1) < self.p:
            img = F.hflip(img)
            heading, distance = target
            target = -heading, 1 - distance
        return img, target

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)
