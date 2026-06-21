import torch

from src.model import SiameseUNet
from src.dataset import DisasterDataset
from src.client import local_train


CLIENTS=[

"earthquake",

"flood",

"wildfire",

"hurricane"

]


for c in CLIENTS:

    print(
        f"\nTRAINING {c}"
    )

    model=SiameseUNet()

    data=DisasterDataset(
        f"data/{c}"
    )

    weights=local_train(

        model,

        data,

        epochs=15

    )

    model.load_state_dict(
        weights
    )

    torch.save(

        model.state_dict(),

        f"{c}_local.pt"

    )

print(
"done"
)