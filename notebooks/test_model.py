import torch

from src.model import SiameseUNet


model=SiameseUNet()

pre=torch.randn(
1,
3,
256,
256
)

post=torch.randn(
1,
3,
256,
256
)

out=model(
pre,
post
)

print(
out.shape
)
