from PIL import Image, ImageDraw
import os
import random

SIZE=256

folders=[
"data/flood/pre",
"data/flood/post",
"data/flood/mask"
]

for f in folders:
    os.makedirs(f,exist_ok=True)

for i in range(50):

    pre=Image.new("RGB",(SIZE,SIZE),"white")
    post=Image.new("RGB",(SIZE,SIZE),"white")

    mask=Image.new("L",(SIZE,SIZE),0)

    d1=ImageDraw.Draw(pre)
    d2=ImageDraw.Draw(post)
    dm=ImageDraw.Draw(mask)

    x=random.randint(40,140)
    y=random.randint(40,140)

    building=[
        x,
        y,
        x+60,
        y+60
    ]

    # PRE
    d1.rectangle(
        building,
        fill="gray"
    )

    damage=1

    # SAME LOCATION
    if damage:

        d2.rectangle(
            building,
            fill="blue"
        )

        dm.rectangle(
            building,
            fill=1
        )

    else:

        d2.rectangle(
            building,
            fill="gray"
        )

    name=f"{i:04d}.png"

    pre.save(
        f"data/flood/pre/{name}"
    )

    post.save(
        f"data/flood/post/{name}"
    )

    mask.save(
        f"data/flood/mask/{name}"
    )

print("done")