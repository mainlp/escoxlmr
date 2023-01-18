#!/bin/bash

for SEED in 276800 381552 497646 624189 884832; do
  for MODEL in escoxlmr xlmr xlmr-mlm; do

    echo "Predicting $MODEL on Green on seed $SEED"
    python3 predict.py \
      logs/green.$MODEL.skills.$SEED/*/model_*.pt \
      data/green/test.conll \
      data/green/$MODEL/green.skills.$SEED.out \
      --dataset green \
      --device 0

  done
done
