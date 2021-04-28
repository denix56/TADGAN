import torch
from NABTraf import NABTraf

if __name__ == '__main__':
    pl_data = NABTraf(batch_size=1)
    pl_data.prepare_data()
    pl_data.setup()
    dl = pl_data.train_dataloader()
    print(next(iter(dl)))
