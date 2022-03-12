import torch
from abc import ABCMeta, abstractmethod
from typing import Dict
from Config.AConfig import AConfig


class AInfoLogger(metaclass=ABCMeta):
    def __init__(self, conf: AConfig):
        self.conf = conf

    @abstractmethod
    def try_print_log(self, loss: torch.Tensor, eval_result: Dict, step: int, global_step: int, epoch_steps: int,
                      epoch: int, num_epochs: int, *args, **kwargs):
        pass
