from src.xbd_geotiff_dataset import XBDGeoDataset
from src.model import SiameseUNet
from src.config import *

import torch
from torch.utils.data import DataLoader
import os


dataset = XBDGeoDataset(
    "data/xbd_full/geotiffs/tier1"
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

model = SiameseUNet().to(
    DEVICE
)

opt = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

bce=torch.nn.BCEWithLogitsLoss(
pos_weight=torch.tensor(
[8.0],
device=DEVICE
)
)

def loss_fn(
pred,
mask
):

 b=bce(
 pred,
 mask
 )

 p=torch.sigmoid(
 pred
 )

 inter=(
 p*mask
 ).sum()

 dice=(
 1-
 (
 2*inter+1
 )
 /
 (
 p.sum()
 +
 mask.sum()
 +
 1
 )
 )

 return (
 0.5*b
 +
 0.5*dice
 )


for epoch in range(EPOCHS):
    model.train()

    total = 0

    for pre,post,mask in loader:

        pre = pre.to(
            DEVICE
        )

        post = post.to(
            DEVICE
        )

        mask = (
            mask
            .unsqueeze(1)
            .to(DEVICE)
        )

        pred = model(
            pre,
            post
        )

        loss = loss_fn(
            pred,
            mask
        )

        opt.zero_grad()

        loss.backward()

        opt.step()

        total += (
            loss.item()
        )
        torch.save(
            model.state_dict(),
            f"{SAVE_DIR}/epoch_{epoch+1}.pt"
)

    print(
        f"epoch {epoch+1}: {total/len(loader):.4f}"
    )


os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    f"{SAVE_DIR}/xbd_real.pt"
)

print("saved")