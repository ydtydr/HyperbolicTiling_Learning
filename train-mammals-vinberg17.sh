#!/bin/bash
#SBATCH -J mm_h2
#SBATCH -o mm_h2.o%j
#SBATCH -e mm_h2.o%j
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=2000
#SBATCH -t 720:00:00
#SBATCH --partition=mpi-cpus  --gres=gpu:0

python3 embed.py \
       -gpu -1 \
       -dim 17 \
       -lr 0.3 \
       -epochs 1000 \
       -negs 50 \
       -burnin 1 \
       -ndproc 4 \
       -polytope vinberg17 \
       -dset wordnet/mammal_closure.csv \
       -batchsize 10 \
       -eval_each 100 \
       -norevery 50 \
       -sparse \
       -train_threads 1 \
       -no-maxnorm

