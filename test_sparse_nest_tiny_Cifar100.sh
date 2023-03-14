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