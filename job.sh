#!/bin/bash

#SBATCH --chdir=/scratch/nas/4/norma/Attacks
#SBATCH --output=/scratch/nas/4/norma/.log/stdout-%j.out
#SBATCH --error=/scratch/nas/4/norma/.log/stderr-%j.out

PYTHON="/scratch/nas/4/norma/venv/bin/python"
CONFIG_FOLDER="/scratch/nas/4/norma/Attacks/Configs" 

for i in {1..12}
do
	$PYTHON ./model.py --config=$CONFIG_FOLDER/config_${i}.json
	mv ./Results/NN.csv ./Results/res_${i}.csv
done
