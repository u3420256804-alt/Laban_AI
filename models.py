import torch.nn as nn
import torch

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden: int, num_layers: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, mask=None):
        out, _ = self.lstm(x)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            out = (out * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        else:
            out = out.mean(dim=1)
        out = self.dropout(out)
        return self.fc(out)

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, p=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=d*(k-1)//2, dilation=d)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=d*(k-1)//2, dilation=d)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(p)
        self.down = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.drop(y)
        return self.relu(y + self.down(x))

class TCNClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int, channels=[256, 256, 256], kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_ch = input_size
        d = 1
        for ch in channels:
            layers.append(TemporalBlock(in_ch, ch, k=kernel, d=d, p=dropout))
            in_ch = ch
            d *= 2
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = self.pool(y).squeeze(-1)
        return self.fc(y)

class TransformerClassifier(nn.Module):
    def __init__(self, input_size: int, num_classes: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        h = self.proj(x)
        if mask is not None:
            key_mask = (mask == 0.0)
            h = self.encoder(h, src_key_padding_mask=key_mask)
        else:
            h = self.encoder(h)
        if mask is not None:
            mask_exp = mask.unsqueeze(-1)
            pooled = (h * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-6)
        else:
            pooled = h.mean(dim=1)
        return self.cls(pooled)