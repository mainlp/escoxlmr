# ESCOXLM-R
___

This repository accompanies the paper: _ESCOXLM-R: Multilingual Taxonomy-driven Pre-training for the Job Market 
Domain_

# Getting Started
___

1. Download the ESCO API from https://esco.ec.europa.eu/en/use-esco/download
2. Run the API locally or where you prefer.
3. Run `get_esco_data.py` to get all relevant data from ESCO.
4. Run `prepare_pretraining_data.py` to prepare the pre-training data for ESCOXLM-R.

Get all packages from `requirements.txt`

# Pre-training ESCOXLM-R
___


Please find the accompanied running scripts in `src/`
* `*-combined.sh` is running both MLM and ERP
* `*-mlm.sh` just runs mlm

# Fine-tuning ESCOXLM-R
___

You can find a working example of MachAmp (https://github.com/machamp-nlp/machamp) in `src/machamp/`
Find in `src/machamp/escoxlmr/` all the scripts to fine-tune the model with MachAmp.

# Licensing
___

This work is licensed under an Apache 2.0 License



