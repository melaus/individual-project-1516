#!/bin/bash

# set the account to be used for the job
#SBATCH --account=free
 
# set name of job
#SBATCH --job-name=prediction_np
#SBATCH --output=prediction_np.out
#SBATCH --error=prediction_np.err
 
# set the number of nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
 
# set max wallclock time
#SBATCH --time=04:00:00
 
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=END
 
# send mail to this address
#SBATCH --mail-user=awll20@bath.ac.uk

# run the application
python prediction.py
