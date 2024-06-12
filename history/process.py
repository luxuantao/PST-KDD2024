import os
from os.path import join
import json
from tqdm import tqdm
from collections import defaultdict as dd
from bs4 import BeautifulSoup
import numpy as np
from fuzzywuzzy import fuzz

import utils
import settings


def prepare_train_test_data_for_glm():
    with open('data/PST/bib_context_train.txt', 'r', encoding='utf-8') as f:
        bib_context_train = f.readlines()
    with open('data/PST/bib_context_valid.txt', 'r', encoding='utf-8') as f:
        bib_context_valid = f.readlines()
    with open('data/PST/bib_context_train_label.txt', 'r', encoding='utf-8') as f:
        train_label = f.readlines()
    with open('data/PST/bib_context_valid_label.txt', 'r', encoding='utf-8') as f:
        valid_label = f.readlines()
    
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    for context, label in zip(bib_context_train, train_label):
        cur_context = "The context is: " + context + ". Is the current reference important? Please answer Yes or No. The answer is [MASK]."
        x_train.append(cur_context)
        y_train.append(label)
    for context, label in zip(bib_context_valid, valid_label):
        cur_context = "The context is: " + context + ". Is the current reference important? Please answer Yes or No. The answer is [MASK]."
        x_valid.append(cur_context)
        y_valid.append(label)

    x_train += x_valid
    y_train += y_valid
    
    print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid))
    
    out_dir = "glm/data/"
    os.makedirs(out_dir, exist_ok=True)

    with open(join(out_dir, "train.json"), "w") as f:
        for i in range(len(x_train)):
            f.write(json.dumps({"inputs_pretokenized": x_train[i], "choices_pretokenized": ["No", "Yes"], "label": y_train[i]}) + "\n")
    
    with open(join(out_dir, "valid.json"), "w") as f:
        for i in range(len(x_valid)):
            f.write(json.dumps({"inputs_pretokenized": x_valid[i], "choices_pretokenized": ["No", "Yes"], "label": y_valid[i]}) + "\n")


if __name__ == "__main__":
    prepare_train_test_data_for_glm()
