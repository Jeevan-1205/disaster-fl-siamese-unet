from torch.utils.data import Dataset
import os
import cv2
import torch


class DisasterDataset(Dataset):

    def __init__(self,root):

        self.pre=os.path.join(root,"pre")
        self.post=os.path.join(root,"post")
        self.mask=os.path.join(root,"mask")

        self.files=sorted(
            os.listdir(self.pre)
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):

        name=self.files[idx]

        pre=cv2.imread(
            os.path.join(
                self.pre,
                name
        )
    )

        post=cv2.imread(
            os.path.join(
            self.post,
            name
        )
    )

        mask=cv2.imread(
            os.path.join(
                self.mask,
                name
            ),
            0
        )


        pre=torch.tensor(
            pre
        ).permute(
            2,
            0,
            1
        ).float()/255.


        post=torch.tensor(
            post
        ).permute(
            2,
            0,
            1
        ).float()/255.


        mask=torch.tensor(
            mask
        ).float()/255.


        return (

        pre,

        post,

        mask

    )