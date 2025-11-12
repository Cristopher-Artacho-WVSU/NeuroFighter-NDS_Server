import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2, dropout: float = 0.2):
        """
        LSTM model for online learning in Dynamic Scripting AI.
        Args:
            input_size (int): Number of input features per timestep.
            hidden_size (int): Number of hidden units in each LSTM layer.
            output_size (int): Number of output features (e.g., adjusted rule weights).
            num_layers (int): Number of stacked LSTM layers.
            dropout (float): Dropout rate between LSTM layers.
        """
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Fully connected layer for output prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Tanh()  # Output between -1 and 1 (for weight adjustment)
        )

    def forward(self, x, hidden=None):
        """
        Forward pass through the network.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            hidden (tuple): Optional tuple of (h_0, c_0) hidden states
        Returns:
            output (Tensor): Model predictions of shape (batch_size, seq_len, output_size)
            hidden (tuple): Updated hidden states
        """
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden


# Example usage (testing)
if __name__ == "__main__":
    model = LSTMNet(input_size=20, hidden_size=64, output_size=5)
    sample_input = torch.randn(1, 5, 8)  # (batch=1, seq_len=5, features=8)
    output, hidden = model(sample_input)
    print("✅ Output shape:", output.shape)
