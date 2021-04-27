import torch
from torch.nn import Module, LSTM, Linear, Sequential, Dropout, Upsample, Conv1d, LeakyReLU, Flatten
from utils import SequenceWise


class Encoder(Module):
    def __init__(self, in_size, ts_size=100, out_size=20, batch_first=False, bias=True):
        super(Encoder, self).__init__()
        self.batch_first = batch_first
        self.hidden_size = 100

        self.rnn = LSTM(input_size=in_size, hidden_size=self.hidden_size, batch_first=batch_first, bidirectional=True,
                        bias=bias)
        self.fc = Linear(2*self.hidden_size*ts_size, out_size, bias=bias)

    def forward(self, x):
        x, _, _ = self.rnn(x)
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        x = torch.flatten(x, 1, -1)
        x = self.fc(x)
        return x


class Generator(Module):
    def __init__(self, in_size=20, bias=True):
        super(Generator, self).__init__()
        self.fc1 = Linear(in_size, 50, bias=bias)
        self.rnn1 = LSTM(input_size=1, hidden_size=64, batch_first=True, bidirectional=True, bias=bias)
        self.dropout1 = Dropout(0.2, True)
        self.up = Upsample(scale_factor=2, mode='nearest')
        self.rnn2 = LSTM(input_size=2*64, hidden_size=64, batch_first=True, bidirectional=True, bias=bias)
        self.dropout2 = Dropout(0.2, True)
        self.fc2 = SequenceWise(Linear(2*64, 1, bias=bias))

    def forward(self, x):
        x = self.fc1(x)
        x.unsqueeze_(-1)
        x, _, _ = self.rnn1(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.up(x)
        x = x.permute(0, 2, 1)
        x, _, _ = self.rnn2(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        return x


# Try spatial conv
class CriticX(Module):
    def __init__(self, in_size=100, bias=True):
        super(CriticX, self).__init__()
        self.net = Sequential(
            Conv1d(in_size, 64, 5, bias=bias),
            LeakyReLU(0.2, True),
            Dropout(0.25, True),
            Conv1d(64, 64, 5, bias=bias),
            LeakyReLU(0.2, True),
            Dropout(0.25, True),
            Conv1d(64, 64, 5, bias=bias),
            LeakyReLU(0.2, True),
            Dropout(0.25, True),
            Conv1d(64, 64, 5, bias=bias),
            LeakyReLU(0.2, True),
            Dropout(0.25, True),
            Flatten(),
            Linear(64, 1, bias=bias)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        return x


class CriticZ(Module):
    def __init__(self, in_size=20, bias=True):
        super(CriticZ, self).__init__()
        self.net = Sequential(
            Linear(in_size, 100, bias=bias),
            LeakyReLU(0.2, True),
            Dropout(0.2, True),
            Linear(100, 100, bias=bias),
            LeakyReLU(0.2, True),
            Dropout(0.2, True),
            Linear(100, 1, bias=bias),
        )

    def forward(self, z):
        z = self.net(z)
        return z