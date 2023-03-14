## Quick Start

### 1. Prepare dataset

* CIFAR: download cifar dataset to folder `~/datasets/cifar` (you may specify this in configuration files).
* ImageNet: download ImageNet dataset to folder `~/datasets/ILSVRC2012` and pre-process with this [script](https://gist.githubusercontent.com/antoinebrl/7d00d5cb6c95ef194c737392ef7e476a/raw/dc53ad5fcb69dcde2b3e0b9d6f8f99d000ead696/prepare.sh).
* Tiny-ImageNet: download Tiny-ImageNet dataset to folder `~/datasets/tiny-imagenet-200`.

### 2. Requirements

* torch>=1.10.2
* torchvision>=0.11.3
* tqdm
* timm
* tensorboard
* scipy
* PyYAML
* pandas
* numpy

### 3. Train from scratch

In dir `config`, we provide some configurations for training, including CIFAR10/100, Tiny-ImageNet and ImageNet.
The following script will start training `NesT-Ti-S (with heavy constituent heads)` from scratch on CIFAR100.

```bash
export CUDA_VISIBLE_DEVICES=0,1

port=9874
python dist_engine.py \
    --num-nodes 1 \
    --rank 0 \
    --master-url tcp://localhost:${port} \
    --backend nccl \
    --multiprocessing \
    --file-name-cfg cls \
    --cfg-filepath config/cifar100/nest/sparse-nest-tiny.yaml \
    --log-dir run/cifar100/nest/sparse-nest-tiny \
    --worker worker
```
> You can train your selected network and dataset by determining the config file in `--cfg-filepath`. We provid all the models used in paper and define them in `\vit_mutual\models\__init__.py`. The prefix of the network is important. Specifically, `"sparse"` indicates the use of single-axis constituents, while `"global-sparse"` indicates employing dual-axis constituents. `"small"` indicates slim heads, and without this prefix means heavy heads.

### 4. Analysis
You can run `visualize_attention.py` to visualize attention, and `hessian.py` to generate Hessian Max Eigenvalues by the power iteration algorithm. `hessian.py` requires PyHessian library that can be installed from pip
```
pip install pyhessian
```