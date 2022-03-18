import os
import sys
import json
import csv
import glob
import pprint
import numpy as np
import random
import argparse
from tqdm import tqdm
from .utils import DataProcessor
from .utils import Coms2SenseSingleSentenceExample
from transformers import (
    AutoTokenizer,
)
from sklearn.metrics import accuracy_score


class Com2SenseDataProcessor(DataProcessor):
    """Processor for Com2Sense Dataset.
    Args:
        data_dir: string. Root directory for the dataset.
        args: argparse class, may be optional.
    """

    def __init__(self, data_dir=None, args=None, **kwargs):
        """Initialization."""
        self.args = args
        self.data_dir = data_dir

        # TODO: Label to Int mapping, dict type.
        self.label2int = {"True": 1, "False": 0}

    def get_labels(self):
        """See base class."""
        return 2  # Binary.

    def _read_data(self, data_dir=None, split="train"):
        """Reads in data files to create the dataset."""
        if data_dir is None:
            data_dir = self.data_dir

        ##################################################
        # TODO: Use json python package to load the data
        # properly.
        # We recommend separately storing the two
        # complementary statements into two individual
        # `examples` using the provided class
        # `Coms2SenseSingleSentenceExample` in `utils.py`.
        # e.g. example_1 = ...
        #      example_2 = ...
        #      examples.append(example_1)
        #      examples.append(example_2)
        # Make sure to add to the examples strictly
        # following the `_1` and `_2` order, that is,
        # `sent_1`'s info should go in first and then
        # followed by `sent_2`'s, otherwise your test
        # results will be messed up!
        # For the guid, simply use the row number (0-
        # indexed) for each data instance, i.e. the index
        # in a for loop. Use the same guid for statements
        # coming from the same complementary pair.
        # Make sure to handle if data do not have
        # labels field.
        json_path = os.path.join(data_dir, split+".json")
        data = json.load(open(json_path, "r"))
        
        examples = list()
        for i, datum in enumerate(data):
            for sent, label in [('sent_1', 'label_1'), ('sent_2', 'label_2')]:
                label_value = None
                if label in datum:
                    label_value = self.label2int[datum[label]]
                examples.append(
                    Coms2SenseSingleSentenceExample(
                        str(i), 
                        datum[sent],
                        label_value,
                        datum['domain'],
                        datum['scenario'],
                        bool(self.label2int[datum['numeracy']])
                    )
                )
        # End of TODO.
        ##################################################

        return examples

    def get_train_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="train")

    def get_dev_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="dev")

    def get_test_examples(self, data_dir=None):
        """See base class."""
        return self._read_data(data_dir=data_dir, split="test")


if __name__ == "__main__":
    # Test loading data.
    proc = Com2SenseDataProcessor(data_dir="datasets/com2sense")
    # train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    # test_examples = proc.get_test_examples()
    # print()
    # for i in range(3):
    #     print(test_examples[i])
    # print()
    
    gt_domains = {}
    gt_scenarios = {}
    gt_numeracy = {True: [], False: []}
    
    pred_domains = {}
    pred_scenarios = {}
    pred_numeracy = {True: [], False: []}

    for i, example in enumerate(val_examples):
        if example.scenario not in gt_scenarios:
            gt_scenarios[example.scenario] = []
            pred_scenarios[example.scenario] = []
        if example.domain not in gt_domains:
            gt_domains[example.domain] = []
            pred_domains[example.domain] = []

    # change this to the location of your prediction file
    #pred_file = "outputs/com2sense/{}/ckpts/com2sense_predictions.txt".format("regular_bert-base-cased_epoch50")
    pred_file = "outputs/com2sense/ckpts/com2sense_predictions.txt"

    predictions = []
    with open(pred_file) as f:
        for line in f:
            predictions.append(int(line.strip()))

    print(predictions[:3])
    for i, example in enumerate(val_examples):
        pred = predictions[i]
        gt = example.label

        #scenario
        gt_scenarios[example.scenario].append(gt)
        pred_scenarios[example.scenario].append(pred)

        #domains
        gt_domains[example.domain].append(gt)
        pred_domains[example.domain].append(pred)

        #numeracy
        gt_numeracy[example.numeracy].append(gt)
        pred_numeracy[example.numeracy].append(pred)

    print("domains:")
    for domain in gt_domains:
        print("{}: {}".format(domain, accuracy_score(gt_domains[domain], pred_domains[domain])))

    print("scenarios:")
    for scenario in gt_scenarios:
        print("{}: {}".format(scenario, accuracy_score(gt_scenarios[scenario], pred_scenarios[scenario])))

    print("numeracy:")
    for boolean in [True, False]:
        print("{}: {}".format(boolean, accuracy_score(gt_numeracy[boolean], pred_numeracy[boolean])))



        
        

    
    
