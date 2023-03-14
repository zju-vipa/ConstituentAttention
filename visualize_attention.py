from visualize import visualize_region_attention, visualize_grid_attention, visualize_grid_attention_v2
from visualize import draw_line_chart
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

import imageio
import torch
from typing import Tuple
from thop import clever_format
from thop import profile
import os
from torchvision import transforms

from vit_mutual.models.transformer import SparseTransformer
from vit_mutual.models.vision_transformers import SparseViT
from vit_mutual.models.vision_transformers.patch_embed import ViTPatchEmbed
from vit_mutual.models.vision_transformers.pos_encoding import PosEncoding, PosEncoding_Learnable
from torchvision import transforms
from my_util import save_image
from vit_mutual.data import build_train_dataset


# helpers
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def run_grid_attention_example(img_path="XXX.jpg", save_path="attention_heat_map/example",
                               attention_mask=None, version=2, quality=100, cnt=1):
    assert version in [1, 2], "We only support two version of attention visualization example"
    if version == 1:
        visualize_grid_attention(img_path=img_path,
                                 save_path=save_path,
                                 attention_mask=attention_mask,
                                 save_image=True,
                                 save_original_image=True,
                                 quality=quality,
                                 cnt=cnt)
    elif version == 2:
        visualize_grid_attention_v2(img_path=img_path,
                                    save_path=save_path,
                                    attention_mask=attention_mask,
                                    save_image=True,
                                    save_original_image=True,
                                    quality=quality,
                                    cnt=cnt)


if __name__ == "__main__":
    img_dir = 'imgnet'
    model_path = 'run/best.pth'

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # transformer = SparseTransformer(embed_dim=288, num_encoder_layers=6, num_heads=9, dim_feedforward=1152, dropout=0.1,
    #                                 activation="gelu", final_norm=True, pre_norm=True)
    # model = SparseViT(num_classes=1000, transformer=transformer, patch_embed=ViTPatchEmbed(embed_dim=288),
    #                   pos_embed=PosEncoding_Learnable(num_tokens=197, embed_dim=288, dropout=0.1)).cuda() 
    # model.load_state_dict(torch.load(model_path)['model'])

    transformer = SparseTransformer(embed_dim=384, num_encoder_layers=12, num_heads=6, dim_feedforward=1536, dropout=0.1,
                                    activation="gelu", final_norm=True, pre_norm=True)
    model = SparseViT(num_classes=1000, transformer=transformer, patch_embed=ViTPatchEmbed(embed_dim=384),
                      pos_embed=PosEncoding_Learnable(num_tokens=197, embed_dim=384, dropout=0.1)).cuda()
    model.load_state_dict(torch.load(model_path)['model'])

    model.eval()
    # cifar: 
    # CIFAR_10_MEAN = [0.4914, 0.4822, 0.4465]
    # CIFAR_10_STD = [0.2023, 0.1994, 0.2010]
    # CIFAR_100_MEAN = [0.5071, 0.4867, 0.4408]
    # CIFAR_100_STD = [0.2023, 0.1994, 0.2010]
    # tiny_imgnet：
    # MEAN = [0.4802, 0.4481, 0.3975]
    # STD = [0.2770, 0.2691, 0.2821]
    # image_net: 
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    def Normalize(data):
        n_max = np.max(data)
        n_min = np.min(data)
        return np.divide(np.subtract(data, n_min), (n_max - n_min))

    for img in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img)
        image = Image.open(img_path)
        image_tensor = transform(image)
        prob, seq_tokens, break_probs, constituent_prior, attns = model(image_tensor.unsqueeze(0).cuda())
        softmax = torch.nn.Softmax(dim=1)(prob)
        scores, predicts = torch.max(softmax, 1)
        cnt = 1
        for attn in attns:
            attn = attn.squeeze(0)
            num_seq = attn.shape[1]
            attn_mean = torch.mean(attn, dim=0).detach().cpu().numpy()
            seq = int(num_seq ** (1 / 2))
            in_attn = np.ones((seq, seq))
            for i in range(seq):
                for j in range(seq):
                    in_attn[i][j] = attn_mean[:, i*seq+j].sum()
            run_grid_attention_example(img_path=img_path, version=2, attention_mask=in_attn,
                                       cnt=cnt)  # version in [1, 2]
            cnt += 1
