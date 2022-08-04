#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    prepare_pretraining_data.py
# @Author:      mikz
# @Time:        27/07/2022 11.36

import json
import pyonmttok
import logging
from datetime import datetime


logging.basicConfig(filename=f"logs/{datetime.now().strftime('%d_%m_%Y_%H:%M:%S')}_pretrain_data.log",
                    filemode="w",
                    format="%(asctime)s::%(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO)


def main():

    langs = ["bg", "es", "cs", "da", "de", "et", "el", "en", "fr", "ga", "hr", "it", "lv", "lt", "hu", "mt", "nl",
             "pl", "pt", "ro", "sk", "sl", "fi", "sv", "is", "no", "ar"]
    tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=False)

    for lang in langs:
        list_of_entities_and_descriptions = []

        cnt = 0
        cnt_desc = 0
        avg_len_descriptions = 0
        avg_len_alt_labels = 0
        avg_len_must_skills = 0
        avg_len_opt_skills = 0

        with open(f"resources/esco_taxonomy/esco_occupations_descriptions_{lang}.json",
                  mode="r+",
                  encoding='utf-8') as f:
            for line in f:
                data = json.loads(line, strict=False)
                description = data["description"].strip()
                tokens = tokenizer(description)

                # gather statistics
                avg_len_descriptions += len(tokens)
                avg_len_alt_labels += len(data["alt_label"] if data.get("alt_label") else [])
                avg_len_must_skills += len(data["must_skills"] if data.get("must_skills") else [])
                avg_len_opt_skills += len(data["opt_skills"] if data.get("opt_skills") else [])
                cnt += 1
                cnt_desc += 1

                list_of_entities_and_descriptions.append({data["esco_code"]: f"{data['pref_label']} {data}"})

                if data.get("alt_label"):
                    for alt_label in data["alt_label"]:
                        list_of_entities_and_descriptions.append({data["esco_code"]: f"{alt_label} {data}"})

                if data.get("must_skills"):
                    for must_skill in data["must_skills"]:
                        tokens_must_skill = tokenizer(must_skill["description"].strip())
                        avg_len_descriptions += len(tokens_must_skill)
                        list_of_entities_and_descriptions.append({data["esco_code"]: f"{must_skill['title']} "
                                                                                     f"{data}"})
                        cnt_desc += 1

                if data.get("opt_skills"):
                    for opt_skill in data["opt_skills"]:
                        tokens_opt_skill = tokenizer(opt_skill["description"].strip())
                        avg_len_descriptions += len(tokens_opt_skill)
                        list_of_entities_and_descriptions.append({data["esco_code"]: f"{opt_skill['title']} "
                                                                                     f"{data}"})
                        cnt_desc += 1

            logging.info(f"current language: {lang}")
            logging.info(f"total entities: {len(list_of_entities_and_descriptions)}")
            logging.info(f"avg len descriptions: {avg_len_descriptions/cnt_desc}")
            logging.info(f"avg len must_skills: {avg_len_must_skills/cnt}")
            logging.info(f"avg len opt_skills: {avg_len_opt_skills/cnt}")

            with open(f"resources/processed/processed_esco_descriptions_{lang}.json", "a+", encoding="utf-8") as fw:
                for item in list_of_entities_and_descriptions:
                    fw.write(json.dumps(item, ensure_ascii=False))
                    if not item == list_of_entities_and_descriptions[-1]:
                        fw.write("\n")


if __name__ == '__main__':
    main()
