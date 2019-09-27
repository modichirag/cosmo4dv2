#! /bin/bash



export OMP_NUM_THREADS=1


afin=1.0000
bs=200
nc=256
B=2
Tsteps=10

echo ${SLURM_NNODES}

##
for seed in {500..10000..100}; do
    output="/global/cscratch1/sd/chmodi/cosmo4d/data/traindata/L$bs-N$nc-B$B-T$Tsteps/S$seed/"
    echo $output
    time srun -N ${SLURM_NNODES}   /global/project/projectdirs/m3127/codes/fd0f424/fastpm/src/fastpm traindata.lua  $bs $nc $seed  $afin $output  $B  $Tsteps
done
