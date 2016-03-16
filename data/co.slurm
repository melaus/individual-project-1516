#!/bin/bash

# set the account to be used for the job
#SBATCH --account=free
 
# set name of job
#SBATCH --job-name=co_1391_1400
#SBATCH --output=co_1391_1400.out
#SBATCH --error=co_1391_1400.err
 
# set the number of nodes
#SBATCH --nodes=1
 
# set max wallclock time
#SBATCH --time=00:40:00
 
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=END
 
# send mail to this address
#SBATCH --mail-user=awll20@bath.ac.uk

# load virtualenv 
source activate py2

# run the application
python transform_data.py -fn co -dim 15 -img_s 1391 -img_e 1400
