from datasets import load_dataset
import numpy as np

def concatenate_fields(example):
    example['model_input'] = example['title'] + "\n" + example['patient']
    return example


def assemble_dataset_for_medium_article():

    nsample = 250
    dataset = load_dataset("aisc-team-b1/PMC-Patients")
    random_state = np.random.RandomState(42)
    chosen_data = random_state.randint(0, len(dataset["train"]), nsample)
    filtered_dataset = dataset["train"].select(chosen_data).select_columns(['patient_id', 'title', 'patient'])
    filtered_dataset = filtered_dataset.map(concatenate_fields)
    return filtered_dataset