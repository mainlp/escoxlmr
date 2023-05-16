# ESCOXLM-R

This repository accompanies the paper: 

__ESCOXLM-R: Multilingual Taxonomy-driven Pre-training for the Job Market Domain__

Mike Zhang, Rob van der Goot, and Barbara Plank. In ACL (2023).

If you use this work please cite the following:

```
<citation coming soon>
```

# I just want to use the pre-trained model

Our pre-trained model can be found on 🤗 https://huggingface.co/jjzha/esco-xlm-roberta-large

# Getting Started

## Requirements

❗Clone the repository. If you use `conda`, install the environment by:

```
conda env create -f environment.yml
```

## ESCO API

❗Next install the ESCO API:

1. Download the ESCO API from https://esco.ec.europa.eu/en/use-esco/download
2. Run the API locally or where you prefer.
3. Run `get_esco_data.py` to get all relevant data from ESCO.
4. Run `prepare_pretraining_data.py` to prepare the pre-training data for ESCOXLM-R.

# Pre-training ESCOXLM-R

❗Note that you need the ESCO data from above.
Please find the accompanied running scripts in `src/`
* `*-combined.sh` is running both MLM and ERP
* `*-mlm.sh` just runs MLM

# Fine-tuning ESCOXLM-R

You can find a working example of MachAmp (https://github.com/machamp-nlp/machamp) in `src/machamp/`
Find in `src/machamp/escoxlmr/` all the scripts to fine-tune the model with MachAmp.

# Licensing

This work is licensed under an Apache 2.0 License
