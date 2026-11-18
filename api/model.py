
import torch.nn as nn


class APTForecaster(nn.Module):
    def __init__(self, i=5, h=128, n=2, p=0.4):
        super().__init__()
        self.lstm = nn.LSTM(i, h, n, batch_first=True, dropout=p, bidirectional=True)
        self.ln = nn.LayerNorm(h * 2)
        self.fc_dropout = nn.Dropout(p)
        self.fc1 = nn.Linear(h * 2, h)
        self.relu = nn.ReLU()
        self.fc2_dropout = nn.Dropout(p)
        self.fc2 = nn.Linear(h, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        o = self.ln(o[:, -1, :])
        o = self.relu(self.fc1(self.fc_dropout(o)))
        return self.fc2(self.fc2_dropout(o))
