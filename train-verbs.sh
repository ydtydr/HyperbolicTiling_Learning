#!/bin/bash
#SBATCH -J vsh2_5
#SBATCH -o vsh2_5.o%j
#SBATCH -e vsh2_5.o%j
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --mem=5000
#SBATCH -t 720:00:00
#SBATCH --partition=mpi-cpus  --gres=gpu:0

DIMS="5"
MODEL="Halfspace"
COMN="1"

while true; do
  case "$1" in
    -c | --com_n ) COMN=$2; shift; shift ;;
    -d | --dim ) DIMS=$2; shift; shift ;;
    -m | --model ) MODEL=$2; shift; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

USAGE="usage: ./train-nouns.sh -d <dim> -m <model>
  -d: dimensions to use
  -m: model to use (can be lorentz or poincare)
  Example: ./train-nouns.sh -m lorentz -d 10
"

case "$MODEL" in
  "Lorentz" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;
  "NLorentz" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;
  "LTiling_rsgd" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;
  "NLTiling_rsgd" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;
  "LTiling_sgd" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;
  "HTiling_rsgd" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;  
  "Halfspace" ) EXTRA_ARGS=("-lr" "0.5" "-no-maxnorm");;  
  "Poincare" ) EXTRA_ARGS=("-lr" "0.5");;
  * ) echo "$USAGE"; exit 1;;
esac

python3 embed.py \
  -dset wordnet/verb_closure.csv \
  -epochs 1000 \
  -negs 50 \
  -burnin 20 \
  -dampening 0.75 \
  -ndproc 4 \
  -eval_each 300 \
  -sparse \
  -burnin_multiplier 0.01 \
  -neg_multiplier 0.1 \
  -lr_type constant \
  -train_threads 5 \
  -dampening 1.0 \
  -batchsize 10 \
  -manifold "$MODEL" \
  -dim "$DIMS" \
  -com_n "$COMN" \
  "${EXTRA_ARGS[@]}"