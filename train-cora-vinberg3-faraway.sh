#!/bin/bash
#SBATCH -J gr_gh2
#SBATCH -o gr_gh2.o%j
#SBATCH -e gr_gh2.o%j
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=5000
#SBATCH -t 720:00:00
#SBATCH --partition=mpi-cpus  --gres=gpu:0

python3 embed.py \
       -dim 3 \
       -lr 0.3 \
       -epochs 1000 \
       -negs 50 \
       -burnin 10 \
       -ndproc 4 \
       -polytope vinberg3 \
       -dset cora/cora_closure.csv \
       -batchsize 50 \
       -eval_each 100 \
       -train_threads 1 \
       -sparse \
       -no-maxnorm \
       -faraway 10000
