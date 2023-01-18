#!/bin/bash

for SEED in 276800 381552 497646 624189 884832; do
  for MODEL in escoxlmr xlmr xlmr-mlm camembert; do

    echo "Predicting $MODEL on fijo on seed $SEED"
    python3 predict.py \
      logs/fijo.$MODEL.softskill.$SEED/*/model_*.pt \
      data/fijo/test.conll \
      data/fijo/$MODEL/fijo.$SEED.out \
      --dataset fijo \
      --device 0

  done
done
