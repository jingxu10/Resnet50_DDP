#PBS -l nodes=4:ppn=2
cd $PBS_O_WORKDIR
mpirun  -machinefile $PBS_NODEFILE ./resnet_ddp.sh
