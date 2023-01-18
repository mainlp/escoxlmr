#!/bin/bash

MODEL_PATH="/data/mike/sayfullina"

for SEED in 276800 381552 497646 624189 884832; do
  for MODEL in escoxlmr xlmr xlmr-mlm; do

    echo "Predicting $MODEL on sayfullina on seed $SEED"
    python3 predict.py \
      $MODEL_PATH/sayfullina.$MODEL.skills.$SEED/*/model_*.pt \
      data/sayfullina/sayfullina_test.conll \
      data/sayfullina/$MODEL/sayfullina.$SEED.out \
      --dataset sayfullina \
      --device 0

  done
done
