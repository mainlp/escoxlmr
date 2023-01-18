#!/bin/bash

for SEED in 276800 381552 497646 624189 884832; do
  for MODEL in escoxlmr xlmr xlmr-mlm; do

    echo "Predicting $MODEL on jobstack on seed $SEED"
    python3 predict.py \
      logs/jobstack.$MODEL.deidentify.$SEED/*/model_*.pt \
      data/jobstack/test.conll \
      data/jobstack/$MODEL/jobstack.$SEED.out \
      --dataset jobstack \
      --device 0

  done
done
