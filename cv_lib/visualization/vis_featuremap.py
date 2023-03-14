from typing import Tuple
from math import sqrt
from PIL import Image
import torch
from my_util import *
import torchvision.transforms.functional as TF


__all__ = [
    "vis_featuremap",
    "vis_seq_token"
]


def vis_featuremap(
        feat: torch.Tensor,
        fp: str,
        n_sigma: float = 3,
        padding: int = 1,
        scale_each: bool = False,
        n_row: int = None,
        **grid_kwargs
) -> torch.Tensor:
    """
    Visualize featuremap of CNN

    Args:
        feat: input featuremap with shape [C, W, H]
        n_sigma: clamp outliners as Gaussian distribution to +/- n_sigma
        save_fp: filepath for save

    Return: `Tensor` with shape [3, W', H']
    """
    assert n_sigma > 0
    # add 0 dim
    feat = feat.unsqueeze(1)
    # normalize x to normal distribution
    feat = (feat - feat.mean()) / feat.std()
    # clip to +/- n_sigma
    feat.clamp_(-n_sigma, n_sigma)
    if n_row is None:
        n_row = int(sqrt(feat.shape[0]) + 0.5)

    # print("feat的形状: ", feat.shape)

    gird_img = make_grid(
        tensor=feat,
        nrow=n_row,
        padding=padding,
        scale_each=scale_each,
        **grid_kwargs
    )
    # print("gird_img.shape: ", gird_img.shape)
    ndarr = gird_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # print("ndarr.shape: ", ndarr.shape)
    # print("ndarr[0]: ", ndarr[:, :, 0])
    # print("ndarr[1]: ", ndarr[:, :, 1])
    # print("ndarr[2]: ", ndarr[:, :, 2])
    """
    im = Image.fromarray(ndarr)
    im.save(fp, format=None)
    """
    return ndarr


def vis_seq_token(
        seq: torch.Tensor,
        feat_shape: Tuple[int, int],
        fp: str,
        n_sigma: float = 3,
        padding: int = 1,
        scale_each: bool = False,
        n_row: int = None,
        **vis_kwargs
) -> torch.Tensor:
    """
    Visualize sequence of tokens of Transformer

    Args:
        seq: input sequence of tokens with shape [N, dim]
        feat_shape: shape of image corresponding to sequence
        n_sigma: clamp outliners as Gaussian distribution to +/- n_sigma
        save_fp: filepath for save

    Return: `Tensor` with shape [3, W', H']
    """
    # make seq to [dim, W, H]
    # print("seq.permute(1, 0)  shape: ", seq.permute(1, 0).shape)
    seq = seq.permute(1, 0).unflatten(dim=-1, sizes=feat_shape)
    res = vis_featuremap(
        feat=seq,
        fp=fp,
        n_sigma=n_sigma,
        padding=padding,
        scale_each=scale_each,
        n_row=n_row,
        **vis_kwargs
    )
    return res
