#!/bin/bash
#SBATCH -J ns_g2
#SBATCH -o ns_g2.o%j
#SBATCH -e ns_g2.o%j
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --mem=5000
#SBATCH -t 720:00:00
#SBATCH --partition=mpi-cpus  --gres=gpu:0
#SBATCH --nodelist=mimno-cpu-01

DIMS="2"
MODEL="group_rie"

while true; do
  case "$1" in
    -d | --dim ) DIMS=$2; shift; shift ;;
    -m | --model ) MODEL=$2; shift; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

case "$MODEL" in
  "lorentz" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;
  "lorentz_product" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;
  "group_rie" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;
  "group_rie_high" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;
  "group_euc" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;
  "halfspace_rie" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;  
  "poincare" ) EXTRA_ARGS=("-lr" "0.5");;
  * ) echo "$USAGE"; exit 1;;
esac

python3 embed.py \
  -dset wordnet/noun_closure.csv \
  -epochs 1000 \
  -negs 50 \
  -burnin 20 \
  -dampening 0.75 \
  -ndproc 4 \
  -eval_each 100 \
  -sparse \
  -burnin_multiplier 0.01 \
  -neg_multiplier 0.1 \
  -lr_type constant \
  -train_threads 5 \
  -dampening 1.0 \
  -batchsize 50 \
  -manifold "$MODEL" \
  -dim "$DIMS" \
  "${EXTRA_ARGS[@]}"