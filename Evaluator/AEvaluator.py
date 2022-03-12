"""
评估器的积累
"""
from abc import ABCMeta, abstractmethod
from typing import Dict, Union
from torch.utils.data import DataLoader
from Model.AModel import AModel
from Config.AConfig import AConfig


class AEvaluator(metaclass=ABCMeta):
    def __init__(self, conf: AConfig):
        self.conf = conf

    @abstractmethod
    def evaluate(self, model: AModel, data_path: str, corpus_path: str, *args, **kwargs) -> Dict:
        pass

    @abstractmethod
    def try_evaluate(self, model: AModel, data_path: str, corpus_path: str, step: int, global_step: int,
                     epoch_steps: int,
                     num_epochs: int, *args, **kwargs) -> Union[Dict, None]:
        pass
