import torch
from torch.utils.data import DataLoader

from src.model import SiameseUNet
from src.dataset import DisasterDataset
from torch.utils.data import ConcatDataset

eq=DisasterDataset(
"data/earthquake"
)

fd=DisasterDataset(
"data/flood"
)

dataset=ConcatDataset(
[
eq,
fd
]
)

loader=DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

model=SiameseUNet()

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


def loss_fn(
    pred,
    target
):

    bce=torch.nn.functional.binary_cross_entropy_with_logits(
        pred,
        target
    )

    pred=torch.sigmoid(
        pred
    )

    smooth=1

    inter=(
        pred*
        target
    ).sum()

    union=(
        pred+
        target
    ).sum()

    dice=(
        2*inter+
        smooth
    )/(
        union+
        smooth
    )

    return (
        0.5*bce+
        0.5*(1-dice)
    )

EPOCHS=15


for epoch in range(
    EPOCHS
):

    total=0

    for pre,post,mask in loader:

        mask=(
            mask
            .float()
            .unsqueeze(1)
        )

        pred=model(
            pre,
            post
        )

        loss=loss_fn(
            pred,
            mask
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total+=loss.item()

    print(
        f"Epoch {epoch+1}: {total:.4f}"
    )


torch.save(
    model.state_dict(),
    "siamese_unet.pt"
)

print(
"saved"
)