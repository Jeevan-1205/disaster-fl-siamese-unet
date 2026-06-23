from torch.utils.data import Dataset
import os
import json
import cv2
import rasterio
import numpy as np
import torch
from shapely import wkt
from src.config import IMG_SIZE


class XBDGeoDataset(Dataset):

    def __init__(self, root):

        self.img_dir = root + "/images"
        self.lbl_dir = root + "/labels"

        files = sorted(
            [
                f for f in os.listdir(self.img_dir)
                if "_pre_disaster.tif" in f
            ]
        )

        self.ids = [
            x.replace(
                "_pre_disaster.tif",
                ""
            )
            for x in files
        ]

    def __len__(self):

        return len(self.ids)

    def load_tif(self,path):

        with rasterio.open(path) as src:

            img = src.read()

        img = np.moveaxis(
            img,
            0,
            -1
        )

        return img

    def create_mask(
        self,
        json_path
    ):

        mask = np.zeros(
            (
                1024,
                1024
            ),
            dtype=np.uint8
        )

        with open(json_path) as f:

            data = json.load(f)

        polys = data[
            "features"
        ][
            "xy"
        ]

        for obj in polys:

            try:

                poly = wkt.loads(
                    obj["wkt"]
                )

                pts = np.array(
                    poly.exterior.coords,
                    dtype=np.int32
                )

                cv2.fillPoly(
                    mask,
                    [pts],
                    1
                )

            except:
                pass

        return mask

    def __getitem__(
        self,
        idx
    ):

        name = self.ids[idx]

        pre = self.load_tif(
            f"{self.img_dir}/{name}_pre_disaster.tif"
        )

        post = self.load_tif(
            f"{self.img_dir}/{name}_post_disaster.tif"
        )

        mask = self.create_mask(
            f"{self.lbl_dir}/{name}_post_disaster.json"
        )

        pre = cv2.resize(
            pre,
            (
                IMG_SIZE,
                IMG_SIZE
            )
        )

        post = cv2.resize(
            post,
            (
                IMG_SIZE,
                IMG_SIZE
            )
        )

        mask = cv2.resize(
            mask,
            (
                IMG_SIZE,
                IMG_SIZE
            )
        )

        pre = (
            torch.tensor(pre)
            .permute(2,0,1)
            /255
        )

        post = (
            torch.tensor(post)
            .permute(2,0,1)
            /255
        )

        mask = torch.tensor(
            mask
        ).float()

        return pre.float(),post.float(),mask