import math
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import TensorDataset, DataLoader, get_worker_info
import pytorch_lightning as pl
from timeseries_preprocessing import time_segments_aggregate, rolling_window_sequences
from data import load_signal, load_anomalies

class TensorDataset2(TensorDataset):
    def set_range(self, start, end):
        self.tensors = [tensor[start:end] for tensor in self.tensors]


class NABTraf(pl.LightningDataModule):
    def __init__(self, data_path: str = 'speed_6005', batch_size: int = 32, num_workers: int = 4):
        super(NABTraf, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.df = None
        self.anomalies = None
        self.data_train = None
        self.data_val = None
        self.index = None
        self.X = None
        self.y = None
        self.X_index = None
        self._dims = (100, 1)

    def setup(self, stage = None):
        data = load_signal(self.data_path)
        anomalies = load_anomalies(self.data_path)
        values, index = time_segments_aggregate(data, interval=1800, time_column='timestamp')
        sk_pipe = Pipeline([('imputer', SimpleImputer()), ('scaler', MinMaxScaler(feature_range=(-1, 1)))])
        values = sk_pipe.fit_transform(values)
        X, self.y, X_index, target_index = rolling_window_sequences(values, index, target_column=0, window_size=100, target_size=1, step_size=1)
        
        self.df = data
        self.anomalies = anomalies
        self.X = torch.tensor(X, dtype=torch.float)
        self.index = torch.tensor(index)
        self.X_index = torch.tensor(X_index)
        self.data_train = TensorDataset(self.X)
        self.data_val = TensorDataset(self.X, self.X_index)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=True)

    def val_dataloader(self):        
        return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=True, persistent_workers=False)



