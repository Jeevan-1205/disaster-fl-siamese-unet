from src.xbd_geotiff_dataset import XBDGeoDataset
from src.model import SiameseUNet
from src.config import *

import matplotlib.pyplot as plt
import torch


dataset=XBDGeoDataset(
"data/xbd_full/geotiffs/tier1"
)

idx=100

pre,post,mask=dataset[idx]

model=SiameseUNet()

model.load_state_dict(
torch.load(
"checkpoints/xbd_real.pt",
map_location=DEVICE
)
)

model.to(
DEVICE
)

model.eval()

with torch.no_grad():

 pred=model(
 pre.unsqueeze(0).to(DEVICE),
 post.unsqueeze(0).to(DEVICE)
 )

 pred=torch.sigmoid(
 pred
 )[0][0]

 pred=(pred>0.5).cpu()


fig,ax=plt.subplots(
1,
4,
figsize=(18,6)
)

ax[0].imshow(
pre.permute(
1,2,0
)
)

ax[0].set_title(
"PRE"
)

ax[1].imshow(
post.permute(
1,2,0
)
)

ax[1].set_title(
"POST"
)

ax[2].imshow(
mask
)

ax[2].set_title(
"GT"
)

ax[3].imshow(
pred
)

ax[3].set_title(
"PRED"
)

plt.show()