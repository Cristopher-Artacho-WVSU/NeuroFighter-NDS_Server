# lstm_model.py
import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, max(8, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden_size // 2), output_size),
            nn.Tanh()  # outputs between -1 and 1 => useful as delta sign/magnitude
        )

    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len, input_size)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# quick test when executed directly
if __name__ == "__main__":
    m = LSTMNet(input_size=20, hidden_size=64, output_size=5)
    sample = torch.randn(1, 1, 20)
    out, h = m(sample)
    print("output shape:", out.shape)
