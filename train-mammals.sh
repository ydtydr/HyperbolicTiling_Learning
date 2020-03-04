#!/bin/bash
#SBATCH -J mmh
#SBATCH -o mmh.o%j
#SBATCH -e mmh.o%j
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=2000
#SBATCH -t 24:00:00
#SBATCH --partition=mpi-cpus  --gres=gpu:0

python3 embed.py \
       -dim 2 \
       -lr 0.3 \
       -epochs 100 \
       -negs 50 \
       -burnin 20 \
       -ndproc 4 \
       -manifold HTiling_rsgd \
       -dset wordnet/mammal_closure.csv \
       -batchsize 10 \
       -eval_each 100 \
       -sparse \
       -train_threads 2

