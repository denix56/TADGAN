import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from timeseries_preprocessing import time_segments_aggregate, rolling_window_sequences
from data import load_signal


class NABTraf(pl.LightningDataModule):
    def __init__(self, data_path: str = 'speed_6005', batch_size: int = 32, num_workers: int = 4):
        super(NABTraf, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.values = None
        self.index = None

    def prepare_data(self):
        data = load_signal(self.data_path)
        values, index = time_segments_aggregate(data, interval=600, time_column='timestamp', method = 'mean')
        sk_pipe = Pipeline([('imputer', SimpleImputer()), ('scaler', MinMaxScaler(feature_range=(-1, 1)))])
        values = sk_pipe.fit_transform(values)
        values = rolling_window_sequences(values, target_column=0, window_size=100, target_size=1)
        self.values = torch.tensor(values)
        self.index = torch.tensor(index)

    def setup(self, stage = None):
        self.data = TensorDataset(self.values, self.index)

    def train_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=True)



