from Config.AConfig import AConfig
from torch.nn import Module
from typing import Dict


class ALossModel(Module):
    def __init__(self, conf: AConfig):
        super().__init__()
        self.conf = conf

    def forward(self, model_output: Dict, ipt: Dict):
        pass
