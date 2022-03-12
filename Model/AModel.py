import torch
from os.path import join


class AModel(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.device = None

    def load(self, load_dir):
        self.load_state_dict(torch.load(join(load_dir, "model_weight.bin"), map_location="cpu"))

    def get_device(self):
        if self.device is None:
            for v in self.state_dict().values():
                self.device = v.device
                break
        return self.device
