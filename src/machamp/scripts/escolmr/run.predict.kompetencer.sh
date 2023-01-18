#!/bin/bash

for SEED in 276800 381552 497646 624189 884832; do
  for MODEL in xlmr xlmr-mlm; do
    for PARAMETERS in enda; do
      for DATASET in en da; do

        echo "Predicting $MODEL on $DATASET on seed $SEED"
        python3 predict.py \
          logs/kompetencer.$MODEL.$PARAMETERS.$SEED/*/model_*.pt \
          data/kompetencer/${DATASET}_test.tsv \
          data/kompetencer/$MODEL/$DATASET.$PARAMETERS.$SEED.out \
          --dataset $DATASET \
          --device 0

      done
    done
  done
done
