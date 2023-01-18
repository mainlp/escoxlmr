#!/bin/bash

MODEL_PATH="logs"

for SEED in 276800 381552 497646 624189 884832; do
  for MODEL in escoxlmr xlmr xlmr-mlm jobbert; do
    for PARAMETERS in multi; do
      for DATASET in house tech; do

        echo "Predicting $MODEL on $DATASET on seed $SEED"
        python3 predict.py \
          $MODEL_PATH/skillspan.$MODEL.$PARAMETERS.$SEED/*/model_*.pt \
          data/skillspan/skillspan_${DATASET}_test.conll \
          data/skillspan/$MODEL/$DATASET.$PARAMETERS.$SEED.out \
          --dataset $DATASET \
          --device 0
      done
    done
  done
done
