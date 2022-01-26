from .dummy_data import DummyDataProcessor
from .com2sense_data import Com2SenseDataProcessor
from .semeval_data import SemEvalDataProcessor

from .processors import DummyDataset
from .processors import Com2SenseDataset
from .processors import SemEvalDataset


data_processors = {
    "dummy": DummyDataProcessor,
    "com2sense": Com2SenseDataProcessor,
    "semeval": SemEvalDataProcessor,
}


data_classes = {
    "dummy": DummyDataset,
    "com2sense": Com2SenseDataset,
    "semeval": SemEvalDataset,
}
