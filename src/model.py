import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, inp, out):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inp, out, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(out, out, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):

        return self.block(x)


class SiameseUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.enc1 = ConvBlock(3, 32)

        self.pool = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)

        self.bridge = ConvBlock(128, 128)

        self.up = nn.ConvTranspose2d(
            128,
            64,
            2,
            2
        )

        self.dec = ConvBlock(
            96,
            64
        )

        self.out = nn.Conv2d(
            64,
            1,
            1
        )

    def encoder(self, x):

        skip = self.enc1(x)

        x = self.pool(skip)

        x = self.enc2(x)

        return skip, x


    def forward(
        self,
        pre,
        post
    ):

        s1, e1 = self.encoder(pre)

        s2, e2 = self.encoder(post)

        diff = torch.abs(
            e1 - e2
        )

        x = self.bridge(
            torch.cat(
                [
                    diff,
                    diff
                ],
                dim=1
            )
        )

        x = self.up(x)

        skip = torch.abs(
            s1 - s2
        )

        x = torch.cat(
            [
                x,
                skip
            ],
            dim=1
        )

        x = self.dec(x)

        return self.out(x)