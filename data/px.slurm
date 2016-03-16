#!/bin/bash

# set the account to be used for the job
#SBATCH --account=free
 
# set name of job
#SBATCH --job-name=px_1301_1448
#SBATCH --output=px_1301_1448.out
#SBATCH --error=px_1301_1448.err
 
# set the number of nodes
#SBATCH --nodes=1
 
# set max wallclock time
#SBATCH --time=04:00:00
 
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=END
 
# send mail to this address
#SBATCH --mail-user=awll20@bath.ac.uk

# load conda
source activate py2

# run the application
python transform_data.py -fn per_pixel -dim 15 -img_s 1410 -img_e 1448 
