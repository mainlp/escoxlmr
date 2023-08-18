# ESCOXLM-R

This repository accompanies the paper: 

__ESCOXLM-R: Multilingual Taxonomy-driven Pre-training for the Job Market Domain__

Mike Zhang, Rob van der Goot, and Barbara Plank. In ACL (2023).

If you use this work please cite the following:

```
@inproceedings{zhang-etal-2023-escoxlm,
    title = "{ESCOXLM}-{R}: Multilingual Taxonomy-driven Pre-training for the Job Market Domain",
    author = "Zhang, Mike  and
      van der Goot, Rob  and
      Plank, Barbara",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.662",
    pages = "11871--11890",
    abstract = "The increasing number of benchmarks for Natural Language Processing (NLP) tasks in the computational job market domain highlights the demand for methods that can handle job-related tasks such as skill extraction, skill classification, job title classification, and de-identification. While some approaches have been developed that are specific to the job market domain, there is a lack of generalized, multilingual models and benchmarks for these tasks. In this study, we introduce a language model called ESCOXLM-R, based on XLM-R-large, which uses domain-adaptive pre-training on the European Skills, Competences, Qualifications and Occupations (ESCO) taxonomy, covering 27 languages. The pre-training objectives for ESCOXLM-R include dynamic masked language modeling and a novel additional objective for inducing multilingual taxonomical ESCO relations. We comprehensively evaluate the performance of ESCOXLM-R on 6 sequence labeling and 3 classification tasks in 4 languages and find that it achieves state-of-the-art results on 6 out of 9 datasets. Our analysis reveals that ESCOXLM-R performs better on short spans and outperforms XLM-R-large on entity-level and surface-level span-F1, likely due to ESCO containing short skill and occupation titles, and encoding information on the entity-level.",
}
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
