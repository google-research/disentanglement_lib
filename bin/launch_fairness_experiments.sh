#!/usr/bin/env bash

# This file is used to launch the 100 jobs that use Risk Minimization to solve
# FitzHugh - Nagumo

# Load the right python module
module load python_gpu/3.6.4

# Add the right folder to the PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/cluster/home/gabbati/disentanglement_lib/"

# Loop on the realizations
for n_run in `seq 0 999`;
do
    wget -O ../fairness_output/$n_run.zip "https://storage.googleapis.com/disentanglement_lib/unsupervised_study_v1/$n_run.zip"
    bsub -n 1 -W 04:00 -R "rusage[mem=8192]" python dlib_fairness_cluster $n_run
done

echo All jobs in queue
