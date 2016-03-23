#!/bin/bash

# set the account to be used for the job
#SBATCH --account=free
 
# set name of job
#SBATCH --job-name=co_1351_1447
#SBATCH --output=co_1351_1447.out
#SBATCH --error=co_1351_1447.err
 
# set the number of nodes
#SBATCH --nodes=1
 
# set max wallclock time
#SBATCH --time=02:50:00
 
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=END
 
# send mail to this address
#SBATCH --mail-user=awll20@bath.ac.uk

# load virtualenv 
source activate py2

# run the application
python transform_data.py patches -fn co -dim 15 -img_s 1351 -img_e 1447
