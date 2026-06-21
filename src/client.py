import copy
import torch
from torch.utils.data import DataLoader


def local_train(
    global_model,
    dataset,
    epochs=3
):

    model=copy.deepcopy(
        global_model
    )

    loader=DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    opt=torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    loss_fn=torch.nn.BCEWithLogitsLoss()

    model.train()

    for e in range(epochs):

        last=0

        for pre,post,mask in loader:

            mask=(
                mask
                .float()
                .unsqueeze(1)
            )

            pred=model(
                pre,
                post
            )

            loss=loss_fn(
                pred,
                mask
            )

            opt.zero_grad()

            loss.backward()

            opt.step()

            last=loss.item()

        print(
            f"epoch {e+1}: {last:.4f}"
        )

    return model.state_dict()