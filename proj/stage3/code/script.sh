#!/bin/bash

#SBATCH --job-name ml-classification   ## name that will show up in the queue
#SBATCH --output result.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=10  ## number of tasks (analyses) to run
#SBATCH --cpus-per-task=2  ## the number of threads allocated to each task
#SBATCH --mem-per-cpu=5G  # memory per CPU core
#SBATCH --partition=class  ## the partitions to run in (comma seperated)
#SBATCH --time=1-01:00:00  ## time for analysis (day-hour:min:sec)

## Load modules
module load anaconda
conda activate my_env


## Insert code, and run your programs here (use 'srun').
## cat <<-EOF > "./mprog-${SLURM_JOB_ID}.conf"
##	0 python main.py data/Human_Activity_Recognition_Using_Smartphones_Data.csv svm
##	1 python main.py data/Human_Activity_Recognition_Using_Smartphones_Data.csv svmnonlinear 
##	2 python main.py data/Human_Activity_Recognition_Using_Smartphones_Data.csv decisiontree
## EOF

## srun --multi-prog "./mprog-${SLURM_JOB_ID}.conf"
srun="srun --ntasks=1 --nodes=1 --exclusive " 
##$srun python main.py data/Human_Activity_Recognition_Using_Smartphones_Data.csv svm > results/svm.out &
##$srun python main.py data/Human_Activity_Recognition_Using_Smartphones_Data.csv svmnonlinear > results/svm_non_linear.out &
##$srun python main.py data/Human_Activity_Recognition_Using_Smartphones_Data.csv decisiontree > results/decision_tree.out & 
##$srun python main.py data/Human_Activity_Recognition_Using_Smartphones_Data.csv logisticregression > results/lr.out &
$srun python main.py data/Human_Activity_Recognition_Using_Smartphones_Data.csv knn > results/knn.out &
$srun python main.py data/Human_Activity_Recognition_Using_Smartphones_Data.csv randomforest > results/randomforest.out &
##$srun python main.py data/Human_Activity_Recognition_Using_Smartphones_Data.csv bagging > results/bagging.out &
##$srun python main.py data/Human_Activity_Recognition_Using_Smartphones_Data.csv ensemblevote > results/ensemble.out &
##$srun python main.py data/Human_Activity_Recognition_Using_Smartphones_Data.csv adaboost > results/adaboost.out &
##$srun python main.py data/Human_Activity_Recognition_Using_Smartphones_Data.csv naivebayes > results/naivebayes.out &
wait 

