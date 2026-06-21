import os
import cv2
import numpy as np
import random


ROOT="data"

CLIENTS=[
"earthquake",
"flood",
"wildfire",
"hurricane"
]

SIZE=128
SAMPLES=60


for c in CLIENTS:

    for f in [
        "pre",
        "post",
        "mask"
    ]:

        os.makedirs(
            f"{ROOT}/{c}/{f}",
            exist_ok=True
        )

    for i in range(
        SAMPLES
    ):

        pre=np.ones(
            (
                SIZE,
                SIZE,
                3
            ),
            dtype=np.uint8
        )*220

        post=pre.copy()

        mask=np.zeros(
            (
                SIZE,
                SIZE
            ),
            dtype=np.uint8
        )

        n=random.randint(
            2,
            5
        )

        for _ in range(n):

            x=random.randint(
                10,
                80
            )

            y=random.randint(
                10,
                80
            )

            w=random.randint(
                20,
                40
            )

            h=random.randint(
                20,
                40
            )

            cv2.rectangle(
                pre,
                (x,y),
                (x+w,y+h),
                (120,120,120),
                -1
            )

            if random.random()<0.5:

                color={

                "earthquake":
                (80,80,80),

                "flood":
                (255,0,0),

                "wildfire":
                (0,100,255),

                "hurricane":
                (180,180,180)

                }[c]

                cv2.rectangle(
                    post,
                    (x,y),
                    (x+w,y+h),
                    color,
                    -1
                )

                cv2.rectangle(
                    mask,
                    (x,y),
                    (x+w,y+h),
                    255,
                    -1
                )

        cv2.imwrite(
            f"{ROOT}/{c}/pre/{i}.png",
            pre
        )

        cv2.imwrite(
            f"{ROOT}/{c}/post/{i}.png",
            post
        )

        cv2.imwrite(
            f"{ROOT}/{c}/mask/{i}.png",
            mask
        )

print("dataset ready")