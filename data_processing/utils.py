import csv
import glob
import json
import logging
import os
from enum import Enum
from typing import List, Optional, Union

import tqdm

from dataclasses import dataclass


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


@dataclass
class DummyExample:
    """
    A single training/test example for Dummy Instance.
    """

    guid: str
    text: str
    label: Optional[int] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass
class Coms2SenseSingleSentenceExample:
    """
    A single training/test example for Com2Sense (single statement) Instance.
    """

    guid: str
    text: str
    label: Optional[int] = None
    domain: Optional[str] = None
    scenario: Optional[str] = None
    numeracy: Optional[bool] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass
class SemEvalSingleSentenceExample:
    """
    A single training/test example for Sem-Eval (single statement) Instance.
    """

    guid: str
    text: str
    label: Optional[int] = None
    right_reason1: Optional[str] = None
    right_reason2: Optional[str] = None
    right_reason3: Optional[str] = None
    confusing_reason1: Optional[str] = None
    confusing_reason2: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"
