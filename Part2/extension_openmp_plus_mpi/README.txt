This is the source code for my OpenMP + MPI hybrid submission.
To compile the code, simply do as you would with the main file.
1. Start in a clean environment (e.g. use 'module purge' etc if required).
2. 'source env.sh'
3. 'make'

To run it just uses a normal SLURM / srun script
(e.g. srun --mpi=pmi2 ./d2q9-bgk input_128x128.params obstacles_128x128.dat)
Make sure to select the desired number of nodes, ranks per node and threads per rank using SLURM variables
--nodes, --ntasks-per-node and --cpus-per-task.
In addition to these, it is CRUCIAL that the SLURM script has the following line BEFORE the code is run:
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
This ensures that the correct number of OMP threads are set.
Here is an example of a simple job script that may be used (with additional files and folder paths potentially needing tweaking):

#!/bin/bash

#SBATCH --job-name=d2q9-bgk
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --time=00:15:00
#SBATCH --partition=teach_cpu
#SBATCH --account=COMS031424
#SBATCH --output=extension.out
#SBATCH --exclusive 

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOB_ID
echo This job runs on the following machines:
echo `echo $SLURM_JOB_NODELIST | uniq`

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun --mpi=pmi2 ./extension_openmp_plus_mpi/d2q9-bgk input_128x128.params obstacles_128x128.dat