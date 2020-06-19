#!/bin/bash

source /opt/intel/inteloneapi/pytorch/latest/bin/activate
python resnet_ddp.py --local_rank=${PMI_RANK} --world_size=${PMI_SIZE}
