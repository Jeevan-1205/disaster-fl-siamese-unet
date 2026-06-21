import torch
import matplotlib.pyplot as plt

from src.model import SiameseUNet
from src.dataset import DisasterDataset


model=SiameseUNet()

model.load_state_dict(
    torch.load(
        "siamese_unet.pt",
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


plt.figure(
    figsize=(15,5)
)

plt.subplot(141)

plt.imshow(
pre.squeeze().permute(1,2,0)
)

plt.title(
"PRE"
)


plt.subplot(142)

plt.imshow(
post.squeeze().permute(1,2,0)
)

plt.title(
"POST"
)


plt.subplot(143)

plt.imshow(
mask
)

plt.title(
"TRUE"
)


plt.subplot(144)

plt.imshow(
pred
)

plt.title(
"PRED"
)


plt.savefig(
"siamese_result.png"
)

print(
"saved"
)