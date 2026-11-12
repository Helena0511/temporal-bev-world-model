import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from resworld.nuscenes_io import CAMERAS, NuScenesTwoHistory


class SimpleCameraBEV(nn.Module):

    def __init__(
        self,
        out_channel=128,
        bev_height=20,
        bev_width=20
    ):

        super().__init__()

        self.bev_height=bev_height
        self.bev_width=bev_width
        self.out_channel=out_channel

        self.image_encoder=nn.Sequential(
            nn.Conv2d(3,32,3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,out_channel,3,stride=2,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.AdaptiveAvgPool2d((bev_height,bev_width))
        )

    def forward(self,x):
        # x=[B,6,3,64,96]

        if x.ndim != 5:
            raise ValueError(
                f"expected [B,6,3,H,W], got {tuple(x.shape)}"
            )

        batch_size, num_cameras, channels, height, width = x.shape
        if num_cameras != 6 or channels != 3:
            raise ValueError(
                "expected six RGB cameras with shape [B,6,3,H,W]"
            )

        x=x.reshape(batch_size * num_cameras, channels, height, width)
        x=self.image_encoder(x)
        x=x.reshape(
            batch_size,
            num_cameras,
            self.out_channel,
            self.bev_height,
            self.bev_width,
        )
        x=x.mean(dim=1)
        x=x.permute(0,2,3,1)

        return x


def load_image(path, image_size=(96, 64)):
    """Read one RGB image and return a normalized [3, H, W] tensor."""
    image = Image.open(path).convert("RGB")
    image = image.resize(image_size, Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()

    # These are the ResWorld config values converted from [0, 255] to [0, 1].
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def load_views(paths, image_size=(96, 64)):
    """Read six camera images and return [6, 3, H, W]."""
    if len(paths) != 6:
        raise ValueError(f"expected six camera paths, got {len(paths)}")
    return torch.stack(
        [load_image(path, image_size=image_size) for path in paths],
        dim=0,
    )


def load_history_views(frame_paths, image_size=(96, 64)):
    """Read two history frames and return [2, 6, 3, H, W]."""
    if len(frame_paths) != 2:
        raise ValueError(f"expected two history frames, got {len(frame_paths)}")
    return torch.stack(
        [load_views(paths, image_size=image_size) for paths in frame_paths],
        dim=0,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    dataset = NuScenesTwoHistory("/mnt/ssd0/datasets/nuscenes")
    current_token = dataset.find_sample_with_two_history()
    three_frames = dataset.get_three_frame_paths(current_token)

    history_views = load_history_views(three_frames[:2])
    current_views = load_views(three_frames[2])
    history_images = history_views.unsqueeze(0)
    current_images = current_views.unsqueeze(0)

    model=SimpleCameraBEV(
        out_channel=128,
        bev_height=20,
        bev_width=20,
    )

    history_bev = torch.stack(
        [model(history_images[:, index]) for index in range(2)],
        dim=1,
    )
    current_bev = model(current_images)

    print("history input:", history_images.shape)
    print("current input:", current_images.shape)
    print("history BEV:", history_bev.shape)
    print("current BEV:", current_bev.shape)

    assert history_views.shape == (2, 6, 3, 64, 96)
    assert history_images.shape == (1, 2, 6, 3, 64, 96)
    assert current_images.shape == (1, 6, 3, 64, 96)
    assert history_bev.shape == (1, 2, 20, 20, 128)
    assert current_bev.shape == (1, 20, 20, 128)
    print("shape test passed")
