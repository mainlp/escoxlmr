import sys
from collections import defaultdict
from typing import List, Tuple
import re
import numpy as np


def calculate_f1(gold: List[Tuple], predicted: List[Tuple]) -> Tuple[float, float, float]:
    # Count the number of true positives, false positives, and false negatives
    tp, fp, fn = 0, 0, 0
    for entity, entity_type in predicted:
        if (entity, entity_type) in gold:
            tp += 1
        else:
            fp += 1
    for entity, entity_type in gold:
        if (entity, entity_type) not in predicted:
            fn += 1

    # Calculate precision, recall, and F1
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1


def calculate_f1_for_entities_and_surface_forms(gold_data: List[Tuple], predicted_data: List[Tuple]) -> Tuple[float, float]:
    # Create lists of entities and sets of surface form tuples
    gold_entities = []
    gold_surface_forms = set()
    predicted_entities = []
    predicted_surface_forms = set()
    for entity, entity_type in gold_data:
        gold_entities.append((entity, entity_type))
        gold_surface_forms.add((entity, entity_type))
    for entity, entity_type in predicted_data:
        predicted_entities.append((entity, entity_type))
        predicted_surface_forms.add((entity, entity_type))

    print(f"Gold entities: {len(gold_entities)}")
    print(f"Surface entities: {len(gold_surface_forms)}")
    print(f"Ratio: {len(gold_surface_forms)/len(gold_entities)}")
    # Calculate F1 for entities
    entity_precision, entity_recall, entity_f1 = calculate_f1(gold_entities, predicted_entities)

    # Calculate F1 for surface forms
    surface_form_precision, surface_form_recall, surface_form_f1 = calculate_f1(gold_surface_forms, predicted_surface_forms)

    return entity_f1, surface_form_f1


def parse_conll_file(filename: str, column: int) -> List[List[Tuple[str, str]]]:
    with open(filename, 'r') as f:
        data = []
        sentence = []
        for line in f:
            line = line.strip()
            if line:
                # Extract the token and label from the line
                token = line.split()[0]
                label = line.split()[int(column)]
                sentence.append((token, label))
            else:
                # Add the current sentence to the list of data and start a new sentence
                data.append(sentence)
                sentence = []
        if sentence:
            # Add the final sentence to the list of data
            data.append(sentence)
        return data


def extract_entities(data: List[List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
    entities = []
    for sentence in data:
        entity_buffer = []
        for token, label in sentence:
            if label != 'O':
                # Extract the entity type from the label
                entity_type = re.sub(r'^[BI]\-', '', label)
                entity_buffer.append(token)
            if label == 'O':
                # Flush the entity buffer if we reach the end of an entity
                if entity_buffer:
                    entity = ' '.join(entity_buffer)
                    entities.append((entity, entity_type))
                    entity_buffer = []
        if entity_buffer:
            # Flush the entity buffer if we reach the end of a sentence
            entity = ' '.join(entity_buffer)
            entities.append((entity, entity_type))
            entity_buffer = []
    return entities


def main(model, column):
    house_skill = []  # paths to prediciton output
    tech_skill = []  # paths to prediction output

    gold_path = "../skillspan/skillspan_house_test.conll"
    gold2_path = "../skillspan/skillspan_tech_test.conll"



    entity_f1_avg = []
    surface_f1_avg = []
    for gold_path, path in zip([gold_path, gold2_path], [house_skill, tech_skill]):
        # Load the gold data and predicted data from CoNLL files
        gold_data_sk = extract_entities(parse_conll_file(gold_path, column=column))
        gold_data_kn = extract_entities(parse_conll_file(gold_path, column=int(column)+1))

        for pred_path in path:
            predicted_data_sk = extract_entities(parse_conll_file(pred_path, column=column))
            predicted_data_kn = extract_entities(parse_conll_file(pred_path, column=int(column)+1))

            # Calculate the F1 scores for both measures
            entity_f1, surface_form_f1 = calculate_f1_for_entities_and_surface_forms(gold_data_sk, predicted_data_sk)
            entity_f1_avg.append(entity_f1)
            surface_f1_avg.append(surface_form_f1)
            entity_f1, surface_form_f1 = calculate_f1_for_entities_and_surface_forms(gold_data_kn, predicted_data_kn)
            entity_f1_avg.append(entity_f1)
            surface_f1_avg.append(surface_form_f1)

    print(f'Entity F1: {np.mean(entity_f1_avg):.5f} +- {np.std(entity_f1_avg):.5f}')
    print(f'Surface Form F1: {np.mean(surface_f1_avg):.5f} +- {np.std(surface_f1_avg):.5f}')


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
