from abc import ABCMeta, abstractmethod
from typing import Dict, Union
from Model.AModel import AModel
from Config.AConfig import AConfig


class AModelSaver(metaclass=ABCMeta):
    def __init__(self, conf: AConfig):
        self.conf = conf

    @abstractmethod
    def try_save_model(self, model: AModel, step: int, global_step: int, epoch_steps: int, num_epochs: int,
                       eval_result: Union[Dict, None],
                       *args, **kwargs) -> bool:
        pass
