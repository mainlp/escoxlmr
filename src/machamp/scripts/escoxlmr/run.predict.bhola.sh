#!/bin/bash

for SEED in 276800 381552 497646 624189 884832; do
  for MODEL in escoxlmr xlmr xlmr-mlm; do

    echo "Predicting $MODEL on Bhola on seed $SEED"
    python3 predict.py \
      logs/bhola.$MODEL.xmlc_mrr.$SEED/*/model_*.pt \
      data/bhola/test.tsv \
      data/bhola/$MODEL/bhola.$SEED.out \
      --dataset bhola \
      --device 0

  done
done
