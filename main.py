import torch
from NABTraf import NABTraf

if __name__ == '__main__':
    pl_data = NABTraf()
    dl = pl_data.train_dataloader()
    print(next(iter(dl)))
