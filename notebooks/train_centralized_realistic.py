import torch
from torch.utils.data import ConcatDataset

from src.model import SiameseUNet
from src.dataset import DisasterDataset
from src.client import local_train


datasets=[

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

full=ConcatDataset(
datasets
)

model=SiameseUNet()

weights=local_train(

model,

full,

epochs=15

)

model.load_state_dict(
weights
)

torch.save(

model.state_dict(),

"centralized_realistic.pt"

)

print(
"centralized saved"
)