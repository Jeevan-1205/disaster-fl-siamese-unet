import torch
import matplotlib.pyplot as plt

from src.model import SiameseUNet
from src.dataset import DisasterDataset
from src.metrics import (
    iou,
    dice,
    precision,
    recall
)


model=SiameseUNet()

model.load_state_dict(
    torch.load(
        "fed_prox.pt",
        map_location="cpu"
    )
)
print(
"Model loaded"
)
model.eval()


dataset=DisasterDataset(
    "data/flood"
)

pre,post,mask=dataset[0]

pre=pre.unsqueeze(0)
post=post.unsqueeze(0)


with torch.no_grad():

    pred=model(
        pre,
        post
    )

pred=torch.sigmoid(
    pred
)
mask=(
mask
.float()
.unsqueeze(0)
)

print(
"IoU:",
round(
iou(
pred,
mask
),
3
)
)

print(
"Dice:",
round(
dice(
pred,
mask
),
3
)
)

print(
"Precision:",
round(
precision(
pred,
mask
),
3
)
)

print(
"Recall:",
round(
recall(
pred,
mask
),
3
)
)
raw=pred.clone()

print(
"Raw logits:"
)

print(
raw.min().item(),
raw.max().item()
)

print(
"Min:",
pred.min().item()
)

print(
"Max:",
pred.max().item()
)

print(
"Mean:",
pred.mean().item()
)

pred=pred.squeeze()

print("pre",pre.shape)
print("post",post.shape)
print("mask",mask.shape)
print("pred",pred.shape)

plt.figure(
    figsize=(15,5)
)

plt.subplot(141)

plt.imshow(
pre.squeeze().permute(1,2,0).cpu()
)

plt.title(
"PRE"
)


plt.subplot(142)

plt.imshow(
post.squeeze().permute(1,2,0).cpu()
)

plt.title(
"POST"
)


plt.subplot(143)

plt.imshow(
mask.squeeze().cpu()
)

plt.title(
"TRUE"
)

plt.subplot(144)

plt.imshow(
pred.squeeze().cpu()
)

plt.title(
"FED"
)


plt.savefig(
"siamese_result.png"
)

print(
"saved"
)