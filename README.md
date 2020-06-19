# Demo of resnet50_ddp with PyTorch

Distributed training (Data Parallel) demo with Resnet50.

* If you would like to check how to run this demo in Intel(R) DevCloud, please checkout devcloud branch.

#### How to run
1. Run with torch.distributed.launch script
```bash
python -m torch.distributed.launch --nproc_per_node=2 resnet_ddp.py
```

2. Run with numactl in separate terminals

    In terminal 0:
    ```bash
    ./resnet_ddp.sh 0 2
    ```
    In terminal 1:
    ```bash
    ./resnet_ddp.sh 1 2
    ```

3. Run with MPI
    1. Make sure MPI backend is enabled.
	2. Change backend to "mpi" in resnet_ddp.py.
	3. Run the following command in terminal.
	```bash
	mpirun -np 4 --map-by slot:PE=14 --bind-to core --report-bindings ./resnet_ddp.sh
	```
