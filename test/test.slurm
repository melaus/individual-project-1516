#!/bin/bash

# set the account to be used for the job
#SBATCH --account=free
 
# set name of job
#SBATCH --job-name=ipp_test1
#SBATCH --output=ipp_test1.out
#SBATCH --error=ipp_test1.err
 
# set the number of nodes
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
 
# set max wallclock time
#SBATCH --time=00:05:00
 
# mail alert at start, end and abortion of execution
#SBATCH --mail-type=END
 
# send mail to this address
#SBATCH --mail-user=awll20@bath.ac.uk

# load conda
#module load anaconda/2.3.0
source activate py2

# run the application
ipcluster start -n 2 --engines=MPI &
python test.py
ipcluster stop
