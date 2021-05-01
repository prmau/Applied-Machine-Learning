#!/bin/bash

#SBATCH --job-name ml-classification   ## name that will show up in the queue
#SBATCH --output result1.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --cpus-per-task=2  ## the number of threads allocated to each task
#SBATCH --mem-per-cpu=5G  # memory per CPU core
#SBATCH --partition=interactive  ## the partitions to run in (comma seperated)
#SBATCH --time=1-01:00:00  ## time for analysis (day-hour:min:sec)

## Load modules
module load anaconda
conda activate my_env

## Insert code, and run your programs here (use 'srun').
srun python main.py data/Human_Activity_Recognition_Using_Smartphones_Data.csv svm > result_1.out &
wait
