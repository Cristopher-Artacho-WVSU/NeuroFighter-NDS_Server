import torch
from torch import nn, optim
from lstm_model import LSTMNet
from preprocessing import preprocess_script
import asyncio
import json

# Setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMNet(input_size=100, hidden_size=64, output_size=10).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Async queue for training data
training_queue = asyncio.Queue()

async def train_worker():
    while True:
        raw_data = await training_queue.get()
        x = preprocess_script(raw_data).to(device).unsqueeze(0)
        # Assuming the label/output is part of raw_data
        y = torch.tensor(raw_data.get("target_output", [0]*10), dtype=torch.float32).to(device).unsqueeze(0)
        
        model.train()
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
