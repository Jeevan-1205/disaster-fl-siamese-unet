import torch


def fedavg(
    weights
):

    avg={}

    for key in weights[0]:

        avg[key]=sum(
            w[key]
            for w in weights
        )/len(weights)

    return avg
import copy


def fedavg(
    weights
):

    avg=copy.deepcopy(
        weights[0]
    )

    for k in avg:

        for i in range(
            1,
            len(weights)
        ):

            avg[k]+=weights[i][k]

        avg[k]/=len(
            weights
        )

    return avg