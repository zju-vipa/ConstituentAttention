from typing import Any, Dict

import torch.nn as nn

import cv_lib.classification.models as cv_models
from .vision_transformers import get_vit, get_deit, get_sparse_vit
from .cnn import get_cnn
from timm.models.efficientnet import efficientnet_b2
from timm.models.nest import nest_tiny, nest_small
from timm.models.sparse_nest import nest_tiny as nest_tiny2
from timm.models.sparse_nest import nest_small as sparse_nest_small
from timm.models.only_final_sparse_nest import nest_tiny as only_sparse_nest_tiny
from timm.models.only_final_sparse_nest import nest_small as only_sparse_nest_small
from timm.models.global_sparse_nest import nest_tiny as nest_tiny3
from timm.models.global_sparse_nest import nest_small as global_sparse_nest_small
from timm.models.pit_avg import pit_ti_224 as pit_tiny
from timm.models.pit_avg import pit_s_224 as pit_small
from timm.models.global_sparse_pit_avg import pit_ti_224 as global_sparse_pit_tiny
from timm.models.global_sparse_pit_avg import pit_s_224 as global_sparse_pit_small
from timm.models.vision_transformer import vit_small_patch16_224
from timm.models.sparse_pit_avg import pit_ti_224 as sparse_pit_tiny
from timm.models.sparse_pit_avg import pit_s_224 as sparse_pit_small
from timm.models.cait import cait_xxs24_224
from timm.models.small_sparse_nest import nest_tiny as small_sparse_nest_tiny
from timm.models.small_sparse_nest import nest_small as small_sparse_nest_small
from timm.models.small_global_sparse_nest import nest_tiny as small_global_sparse_nest_tiny
from timm.models.small_global_sparse_nest import nest_small as small_global_sparse_nest_small
# from timm.models.sparse_cait import cait_xxs24_224 as sparse_cait_xxs24_224
# from timm.models.global_sparse_cait import cait_xxs24_224 as global_sparse_cait_xxs24_224
# from timm.models.self_global_sparse_nest import nest_small as self_global_sparse_nest_small
# from timm.models.self_global_sparse_pit_avg import pit_ti_224 as self_global_sparse_pit_tiny

cv_models.register_model("efficientnet_b2", efficientnet_b2)

__REGISTERED_MODELS__ = {
    "sparse_vit": get_sparse_vit,
    "vit": get_vit,
    "deit": get_deit,
    "cnn": get_cnn,
    "nest_tiny": nest_tiny,
    "sparse_nest_tiny": nest_tiny2,
    "global_sparse_nest_tiny": nest_tiny3,
    "pit_tiny": pit_tiny,
    "sparse_pit_tiny": sparse_pit_tiny,
    "global_sparse_pit_tiny": global_sparse_pit_tiny,
    "vit_small": vit_small_patch16_224,
    "cait": cait_xxs24_224,
    "nest_small": nest_small,
    "sparse_nest_small": sparse_nest_small,
    "pit_small": pit_small,
    "sparse_pit_small": sparse_pit_small,
    "global_sparse_nest_small": global_sparse_nest_small,
    # "self_global_sparse_nest_small": self_global_sparse_nest_small,
    # "self_global_sparse_pit_tiny": self_global_sparse_pit_tiny,
    "global_sparse_pit_small": global_sparse_pit_small,
    # "sparse_dis_vit": get_sparse_dis_vit,
    # "sparse_cait": sparse_cait_xxs24_224,
    # "global_sparse_cait": global_sparse_cait_xxs24_224,
    "only_sparse_nest_tiny": only_sparse_nest_tiny,
    "only_sparse_nest_small": only_sparse_nest_small,
    "small_sparse_nest_tiny": small_sparse_nest_tiny,
    "small_sparse_nest_small": small_sparse_nest_small,
    "small_global_sparse_nest_tiny": small_global_sparse_nest_tiny,
    "small_global_sparse_nest_small": small_global_sparse_nest_small,
    "official_models": cv_models.get_model
}


class ModelWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        if isinstance(output, dict):
            return output
        ret = {
            "pred": output
        }
        return ret


def get_model(model_cfg: Dict[str, Any], num_classes: int, with_wrapper: bool = True) -> nn.Module:
    # print("model_cfg: ", model_cfg)
    # exit(0)
    if model_cfg["name"] == "vit" or model_cfg["name"] == "sparse_vit" or model_cfg["name"] == "sparse_dis_vit":
        model = __REGISTERED_MODELS__[model_cfg["name"]](model_cfg, num_classes)
    else:
        model = __REGISTERED_MODELS__[model_cfg["name"]](pretrained=model_cfg["pretrained"], num_classes=num_classes)

    # if model_cfg["name"] == "nest_tiny" or model_cfg["name"] == "sparse_nest_tiny"\
    #         or model_cfg["name"] == "global_sparse_nest_tiny" or model_cfg["name"] == "pit_tiny":
    #     model = __REGISTERED_MODELS__[model_cfg["name"]]()
    # else:
    #     model = __REGISTERED_MODELS__[model_cfg["name"]](model_cfg, num_classes)
    if with_wrapper:
        model = ModelWrapper(model)
    return model
