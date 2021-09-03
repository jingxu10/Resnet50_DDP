# Demo of resnet50_ddp with PyTorch

Distributed training (Distributed Data Parallel) demo with Resnet50.

* If you would like to check how to run this demo in Intel(R) DevCloud, please checkout devcloud branch.

#### How to run
1. Run with torch.distributed.launch script
```bash
python -m torch.distributed.launch --nproc_per_node=2 resnet_ddp.py
```

2. Run with torchrun
```bash
torchrun --nproc_per_node=2 resnet_ddp.py
```

3. Run with IPEX launch script
```bash
source /opt/intel/oneapi/mpi/latest/env/vars.sh
python launch.py --distributed --nproc_per_node 2 resnet_ddp.py
```

4. Run with Horovod
```bash
horovodrun -np 2 python resnet_ddp.py
```

#### Set backend
```bash
python resnet_ddp.py --backend [ccl|nccl|gloo|...]
```
