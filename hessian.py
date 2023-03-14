import numpy as np
import torch 
from torchvision import datasets, transforms
from PyHessian.pyhessian import hessian # Hessian computation
# from density_plot import get_esd_plot # ESD plot
from pytorchcv.model_provider import get_model as ptcv_get_model # model

import matplotlib.pyplot as plt
import os
# pylint: disable = C, R, E1101, E1123
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from PyHessian.density_plot import get_esd_plot # ESD plot
import torchvision
# import os, sys
# sys.path.append('../hessian')

import argparse
from functools import partial
import traceback
import os
import signal
from typing import List

import torch.optim
import torch.multiprocessing as mp

from cv_lib.logger import MultiProcessLoggerListener
from cv_lib.config_parsing import get_eval_logger, get_train_logger, get_cfg
from cv_lib.utils import to_json_str

from vit_mutual.utils import DistLaunchArgs, LogArgs
from vit_mutual.tasks.worker import worker
from vit_mutual.tasks.worker_mutual import mutual_worker
from vit_mutual.tasks.worker_eval import eval_worker
from vit_mutual.tasks.sam_train_worker import sam_train_worker
import os
import logging
import shutil
import time
from logging.handlers import QueueHandler
from typing import Dict, Any, List
import datetime
import yaml
import collections

from torch import nn, Tensor
import torch.cuda
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn
from torch.cuda import amp
from utils.logger import Logger, savefig

import cv_lib.utils as utils
from cv_lib.config_parsing import get_tb_writer
from cv_lib.optimizers import get_optimizer
from cv_lib.schedulers import get_scheduler
import cv_lib.distributed.utils as dist_utils

from vit_mutual.data import build_train_dataset
from vit_mutual.models import get_model, ModelWrapper
from vit_mutual.loss import get_loss_fn, Loss
import vit_mutual.utils as vit_utils
from vit_mutual.eval import Evaluation

import numpy as np
import torch
import easydict
import os, sys
from tqdm import tqdm, trange
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import linear_model

import torch.nn.functional as F
import matplotlib.pylab as plt
import matplotlib.tri as tri
import pickle as pkl
import copy
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from vit_mutual.models.vision_transformers import *
from vit_mutual.models.vision_transformers.vit import ViT
from vit_mutual.models.transformer import SparseTransformer
from vit_mutual.models.vision_transformers import SparseViT
from vit_mutual.models.vision_transformers.patch_embed import ViTPatchEmbed
from vit_mutual.models.vision_transformers.pos_encoding import PosEncoding, PosEncoding_Learnable

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

args = easydict.EasyDict({
    'best_model_path': 'run/hessian/vit-base-iter-XXX.pth',
    'device': torch.device('cuda:0'),
})

global_cfg = get_cfg('config/cifar100/vit/vit-base.yaml')

data_cfg: Dict[str, Any] = global_cfg["dataset"]
train_cfg: Dict[str, Any] = global_cfg["training"]
val_cfg: Dict[str, Any] = global_cfg["validation"]
model_cfg: Dict[str, Any] = global_cfg["model"]
loss_cfg: Dict[str, Any] = global_cfg["loss"]
launch_args =  vit_utils.DistLaunchArgs
launch_args.distributed = False

device = args.device

myloss = get_loss_fn(loss_cfg)
myloss.to(device)


print("Building dataset...")
train_loader, test_loader, n_classes = build_train_dataset(
    data_cfg,
    train_cfg,
    val_cfg,
    launch_args,
)

print(len(train_loader))

print("Building model...")

transformer = Transformer(embed_dim=384, num_encoder_layers=12, num_heads=6, dim_feedforward=1536, dropout=0.1,
                                    activation="gelu", final_norm=True, pre_norm=True)
model = ViT(num_classes=100, transformer=transformer, patch_embed=ViTPatchEmbed(embed_dim=384),
            pos_embed=PosEncoding_Learnable(num_tokens=197, embed_dim=384, dropout=0.1))
model = model.to(device)
model.load_state_dict(torch.load(args.best_model_path)['model'])
n = sum(p.numel() for p in model.parameters())
model.eval()

transformer2 = SparseTransformer(embed_dim=384, num_encoder_layers=12, num_heads=6, dim_feedforward=1536, dropout=0.1,
                                    activation="gelu", final_norm=True, pre_norm=True)
model2 = SparseViT(num_classes=100, transformer=transformer2, patch_embed=ViTPatchEmbed(embed_dim=384),
                      pos_embed=PosEncoding_Learnable(num_tokens=197, embed_dim=384, dropout=0.1))
model2 = model2.to(device)
model2.load_state_dict(torch.load('run/hessian/sparse-vit-base-XXX.pth')['model'])
n = sum(p.numel() for p in model2.parameters())

# change the model to eval mode to disable running stats upate
model2.eval()

# create loss function
criterion = torch.nn.CrossEntropyLoss()

# get dataset 
# train_loader, test_loader = getData()


logger = Logger('hessian_log/vit_base_cifar100.txt', title='vit_base_cifar100')
logger.set_names(['Max Hessian eigenvalue'])

logger2 = Logger('hessian_log/sparse_vit_base_cifar100.txt', title='sparse_vit_base_cifar100')
logger2.set_names(['Max Hessian eigenvalue2'])

for inputs, targets in train_loader:
    inputs, targets = vit_utils.move_data_to_device(inputs, targets, device)
    hessian_comp = hessian(model, criterion, data=(inputs, targets['label']), cuda=True)
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
    for i in range(5):
        logger.append([top_eigenvalues[i]])
    print("top_eigenvalues for vit-base: ", top_eigenvalues)

    hessian_comp2 = hessian(model2, criterion, data=(inputs, targets['label']), cuda=True)
    top_eigenvalues, top_eigenvector = hessian_comp2.eigenvalues()
    for i in range(5):
        logger2.append([top_eigenvalues[i]])
    print("top_eigenvalues for sparse-vit-base: ", top_eigenvalues)