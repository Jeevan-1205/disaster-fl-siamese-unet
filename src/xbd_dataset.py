from torch.utils.data import Dataset
from src.config import IMG_SIZE

import os
import cv2
import torch


class XBDDataset(Dataset):

    def __init__(self, root):

        self.img = os.path.join(
            root,
            "images"
        )

        self.mask = os.path.join(
            root,
            "targets"
        )

        files = sorted(
            os.listdir(
                self.img
            )
        )

        self.ids = sorted(
            list(
                set(
                    [
                        x.replace(
                            "_pre_disaster.png",
                            ""
                        ).replace(
                            "_post_disaster.png",
                            ""
                        )
                        for x in files
                    ]
                )
            )
        )

    def __len__(self):

        return len(
            self.ids
        )

    def __getitem__(
        self,
        idx
    ):

        name = self.ids[idx]

        pre = cv2.imread(
            os.path.join(
                self.img,
                name+"_pre_disaster.png"
            )
        )

        post = cv2.imread(
            os.path.join(
                self.img,
                name+"_post_disaster.png"
            )
        )

        mask = cv2.imread(
            os.path.join(
                self.mask,
                name+"_post_disaster_target.png"
            ),
            0
        )

        pre = cv2.resize(
            pre,
            (IMG_SIZE,IMG_SIZE)
        )

        post = cv2.resize(
            post,
            (IMG_SIZE,IMG_SIZE)
        )

        mask = cv2.resize(
            mask,
            (IMG_SIZE,IMG_SIZE)
        )

        pre = (
            torch.tensor(pre)
            .permute(2,0,1)
            /255.
        )

        post = (
            torch.tensor(post)
            .permute(2,0,1)
            /255.
        )

        mask = (
            torch.tensor(mask)
            >0
        ).float()

        return (
            pre.float(),
            post.float(),
            mask
        )