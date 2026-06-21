import torch

from src.model import SiameseUNet
from src.dataset import DisasterDataset

from src.client_prox import local_train_prox
from src.fedavg import fedavg


clients=[

DisasterDataset(
"data/earthquake"
),

DisasterDataset(
"data/flood"
),

DisasterDataset(
"data/wildfire"
),

DisasterDataset(
"data/hurricane"
)

]


global_model=SiameseUNet()

ROUNDS=5


for r in range(
ROUNDS
):

    print(
    f"\nROUND {r+1}"
    )

    weights=[]

    for i,c in enumerate(
    clients
    ):

        print(
        f"client {i+1}"
        )

        w=local_train_prox(
            global_model,
            c,
            epochs=3
        )

        weights.append(
            w
        )

    global_model.load_state_dict(

        fedavg(
            weights
        )

    )


torch.save(

global_model.state_dict(),

"fed_prox.pt"

)

print(
"training done"
)