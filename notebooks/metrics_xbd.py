from src.xbd_geotiff_dataset import XBDGeoDataset
from src.model import SiameseUNet
from src.config import *

import torch


dataset=XBDGeoDataset(
"data/xbd_full/geotiffs/tier1"
)

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


tp=0
fp=0
fn=0


for i in range(200):

 pre,post,mask=dataset[i]

 with torch.no_grad():

  pred=model(
   pre.unsqueeze(0).to(DEVICE),
   post.unsqueeze(0).to(DEVICE)
  )

 pred=(
 torch.sigmoid(
 pred
 )[0][0]
 >0.3
 ).cpu()

 tp+=(
 pred
 &
 mask.bool()
 ).sum()

 fp+=(
 pred
 &
 ~mask.bool()
 ).sum()

 fn+=(
 ~pred
 &
 mask.bool()
 ).sum()


precision=tp/(tp+fp+1e-8)

recall=tp/(tp+fn+1e-8)

dice=(
2*tp
/
(
2*tp
+
fp
+
fn
+
1e-8
)
)

iou=(
tp
/
(
tp
+
fp
+
fn
+
1e-8
)
)


print(
"IoU:",
float(iou)
)

print(
"Dice:",
float(dice)
)

print(
"Precision:",
float(precision)
)

print(
"Recall:",
float(recall)
)