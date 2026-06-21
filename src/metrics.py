import torch


def binarize(
    pred,
    threshold=0.5
):
    return (
        pred >
        threshold
    ).float()


def iou(
    pred,
    target
):

    pred=binarize(pred)

    inter=(
        pred*target
    ).sum()

    union=(
        pred+
        target
    ).clamp(
        0,
        1
    ).sum()

    return (
        inter/
        (
            union+
            1e-8
        )
    ).item()


def dice(
    pred,
    target
):

    pred=binarize(pred)

    inter=(
        pred*
        target
    ).sum()

    return (
        (
            2*inter
        )
        /
        (
            pred.sum()
            +
            target.sum()
            +
            1e-8
        )
    ).item()


def precision(
    pred,
    target
):

    pred=binarize(pred)

    tp=(
        pred*
        target
    ).sum()

    fp=(
        pred*
        (
            1-target
        )
    ).sum()

    return (
        tp/
        (
            tp+
            fp+
            1e-8
        )
    ).item()


def recall(
    pred,
    target
):

    pred=binarize(pred)

    tp=(
        pred*
        target
    ).sum()

    fn=(
        (
            1-pred
        )
        *
        target
    ).sum()

    return (
        tp/
        (
            tp+
            fn+
            1e-8
        )
    ).item()