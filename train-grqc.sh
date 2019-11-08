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
       -dim 2 \
       -lr 0.3 \
       -epochs 1000 \
       -negs 50 \
       -burnin 20 \
       -ndproc 4 \
       -manifold group_rie \
       -dset wordnet/grqc.csv \
       -batchsize 10 \
       -eval_each 100 \
       -sparse \
       -train_threads 2