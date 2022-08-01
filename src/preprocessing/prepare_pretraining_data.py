#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    prepare_pretraining_data.py
# @Author:      mikz
# @Time:        27/07/2022 11.36

import json
import pyonmttok


def main():

    list_of_entities_and_descriptions = []

    cnt = 0
    cnt_desc = 0
    avg_len_descriptions = 0
    avg_len_alt_labels = 0
    avg_len_must_skills = 0
    avg_len_opt_skills = 0

    tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=False)
    with open("resources/esco_occupations_descriptions_en.json", "r") as f:
        for line in f:
            data = json.loads(line)
            tokens = tokenizer(data["description"])

            # gather statistics
            avg_len_descriptions += len(tokens)
            avg_len_alt_labels += len(data["alt_label"])
            avg_len_must_skills += len(data["must_skills"])
            avg_len_opt_skills += len(data["opt_skills"])
            cnt += 1
            cnt_desc += 1

            list_of_entities_and_descriptions.append({data["esco_code"]: f"{data['pref_label']} {' '.join(tokens)}"})

            for alt_label in data["alt_label"]:
                list_of_entities_and_descriptions.append({data["esco_code"]: f"{alt_label} {' '.join(tokens)}"})

            for must_skill in data["must_skills"]:
                tokens_must_skill = tokenizer(must_skill["description"])
                avg_len_descriptions += len(tokens_must_skill)
                list_of_entities_and_descriptions.append({data["esco_code"]: f"{must_skill['title']} "
                                                                             f"{' '.join(tokens_must_skill)}"})
                cnt_desc += 1

            for opt_skill in data["opt_skills"]:
                tokens_opt_skill = tokenizer(opt_skill["description"])
                avg_len_descriptions += len(tokens_opt_skill)
                list_of_entities_and_descriptions.append({data["esco_code"]: f"{opt_skill['title']} "
                                                                             f"{' '.join(tokens_opt_skill)}"})
                cnt_desc += 1

    print(f"total entities: {len(list_of_entities_and_descriptions)}")
    print(f"avg len descriptions: {avg_len_descriptions/cnt_desc}")
    print(f"avg len must_skills: {avg_len_must_skills/cnt}")
    print(f"avg len opt_skills: {avg_len_opt_skills/cnt}")

    for item in list_of_entities_and_descriptions:
        print(json.dumps(item))

if __name__ == '__main__':
    main()
