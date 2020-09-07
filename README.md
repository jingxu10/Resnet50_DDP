# Demo of resnet50_ddp with PyTorch torch-ccl

Distributed training (Data Parallel) demo with Resnet50 with oneCCL.

#### How to run
1. Setup environment following instructions in (this tutorial)[https://github.com/intel/optimized-models/tree/master/pytorch/distributed]
  * Note: Better to use 762270c commit PyTorch
2. Edit configurations in resnet_ddp.sh according to your device
3. Run resnet_ddp.sh script
    ```bash
    ./resnet_ddp.sh
    ```
