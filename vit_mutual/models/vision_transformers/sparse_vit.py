from typing import List

import torch
import torch.nn as nn

from .patch_embed import PatchEmbed
from .pos_encoding import PosEncoding
from vit_mutual.models.transformer import SparseTransformer
from vit_mutual.models.transformer.sparse_transformer import MLP, SparseMultiHeadSelfAttention, GroupAttention


class SparseViT(nn.Module):
    def __init__(
            self,
            patch_embed: PatchEmbed,
            pos_embed: PosEncoding,
            transformer: SparseTransformer,
            num_classes: int
    ):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = patch_embed
        self.pos_embed = pos_embed
        self.transformer = transformer

        embed_dim = self.transformer.embed_dim
        self.cls_head = nn.Linear(self.transformer.embed_dim, num_classes)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.cls_token, std=0.02)

    def get_mhsa(self) -> List[SparseMultiHeadSelfAttention]:
        mhsa = [layer.attention for layer in self.transformer.layers]
        return mhsa

    def get_mlp(self) -> List[MLP]:
        mlp = [layer.mlp for layer in self.transformer.layers]
        return mlp

    def forward(self, img: torch.Tensor):
        # seq has shape [n, bs, dim]
        seq: torch.Tensor = self.patch_embed(img)
        # pos embedding
        seq = self.pos_embed(seq)
        seq, seq_tokens, break_probs, constituent_prior, attns = self.transformer(seq)
        seq = torch.mean(seq, dim=0)
        prob = self.cls_head(seq)

        return prob, seq_tokens, break_probs, constituent_prior, attns  # , break_probs


"""
    def forward(self, img: torch.Tensor):
        # seq has shape [n, bs, dim]
        seq: torch.Tensor = self.patch_embed(img)
        bs = seq.shape[1]
        cls_token = self.cls_token.expand(-1, bs, -1)
        # add cls token
        seq = torch.cat((cls_token, seq), dim=0)
        # pos embedding
        seq = self.pos_embed(seq)
        seq, break_probs = self.transformer(seq)
        print("seq.shape: ", seq.shape)
        cls_token = seq[0]
        print("cls_token.shape: ", cls_token.shape)
        exit(0)
        prob = self.cls_head(cls_token)

        return prob  # , break_probs
"""
