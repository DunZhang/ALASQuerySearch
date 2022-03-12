import torch
import torch.nn.functional as F

if __name__ == "__main__":
    x1 = torch.rand((15,10,128))
    x2 = torch.rand((15,10,1))
    print(torch.mul(x1,x2).shape)