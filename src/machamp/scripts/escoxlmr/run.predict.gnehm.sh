#!/bin/bash

for SEED in 276800 381552 497646 624189 884832; do
  for MODEL in escoxlmr xlmr xlmr-mlm jobgbert; do

    echo "Predicting $MODEL on Gnehm on seed $SEED"
    python3 predict.py \
      logs/gnehm.$MODEL.ict.$SEED/*/model_*.pt \
      data/gnehm/test.conll \
      data/gnehm/$MODEL/gnehm.ict.$SEED.out \
      --dataset gnehm \
      --device 0

  done
done
