#!/bin/bash
#SBATCH -J mmh2_2
#SBATCH -o mmh2_2.o%j
#SBATCH -e mmh2_2.o%j
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=2000
#SBATCH -t 24:00:00
#SBATCH --partition=mpi-cpus  --gres=gpu:0

python3 embed.py \
       -dim 2 \
       -com_n 1 \
       -lr 0.3 \
       -epochs 1000 \
       -negs 50 \
       -burnin 20 \
       -ndproc 4 \
       -polytope Halfspace \
       -dset wordnet/mammal_closure.csv \
       -batchsize 10 \
       -eval_each 20 \
       -sparse \
       -train_threads 2
