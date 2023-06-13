import os
from pathlib import Path
from sys import stderr

from lightnet.models import Darknet
from lightnet.network.module import Darknet as DarknetBase
from torch.hub import download_url_to_file, get_dir, urlparse

__all__ = ["Darknet", "load_darknet_from_url", "DarknetURL"]


DarknetURL = "http://pjreddie.com/media/files/darknet.weights"
# And Darknet19URL, Darknet53URL, etc.


def load_darknet_from_url(
    module: DarknetBase, url: str, progress: bool = True, filename: str = None
):
    r"""Adapted from torch.hub.load_state_dict_from_url"""

    model_dir = Path(get_dir()) / "checkpoints"
    os.makedirs(model_dir, exist_ok=True)
    parts = urlparse(url)
    filename = filename or Path(parts.path).name
    cached_file = model_dir / filename
    if not cached_file.exists():
        stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, cached_file, None, progress=progress)
    module.load(cached_file.as_posix())
