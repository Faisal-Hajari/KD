set -ex

deepspeed --num_nodes 1 --num_gpus 8 train.py --deepspeed_config "configs/ds_config.json" --deepspeed