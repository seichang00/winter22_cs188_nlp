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
from .utils import DummyExample
from transformers import (
    AutoTokenizer,
)


class DummyDataProcessor(DataProcessor):
    """Processor for Dummy Dataset.
    Args:
        data_dir: string. Root directory for the dataset.
        args: argparse class, may be optional.
    """

    def __init__(self, data_dir=None, args=None, **kwargs):
        """Initialization."""
        self.args = args
        self.data_dir = data_dir

    def get_labels(self):
        """See base class."""
        return 2  # Binary.

    def _read_data(self, data_dir=None, split="train"):
        """Reads in data files to create the dataset."""
        if data_dir is None:
            data_dir = self.data_dir

        json_path = os.path.join(data_dir, split+".json")
        data = json.load(open(json_path, "r"))
        
        examples = []

        for i in range(len(data)):
            datum = data[str(i)]
            guid = datum["id"]
            sentence = datum["sentence"]
            label = datum["label"]

            example = DummyExample(
                guid=guid,
                text=sentence,
                label=label
            )
            examples.append(example)

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
    proc = DummyDataProcessor(data_dir="datasets/dummies")
    train_examples = proc.get_train_examples()
    val_examples = proc.get_dev_examples()
    test_examples = proc.get_test_examples()
    print()
    for i in range(3):
        print(test_examples[i])
    print()
