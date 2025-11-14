# lstm_online.py
import asyncio
import json
import os
import torch
from torch import nn, optim
from preprocessing import preprocess_script
from lstm_model import LSTMNet
from trainer_queue import training_queue

# TRAINER CONFIG (should match server INPUT_SIZE & OUTPUT_SIZE)
INPUT_SIZE = 20
HIDDEN_SIZE = 64
OUTPUT_SIZE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Trainer has its own model copy; it will save to same CACHE_PATH used by server.
CACHE_PATH = "lstm_cached.pth"

model = LSTMNet(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Try load existing model weights if present
if os.path.exists(CACHE_PATH):
    try:
        model.load_state_dict(torch.load(CACHE_PATH, map_location=device))
        print("✅ Trainer loaded cached model.")
    except Exception as e:
        print("⚠️ Trainer failed to load cached model:", e)

model.train()

# Simple helper that turns a (1,1,input_size) x into a pseudo-target
# For online/self-supervised update we use identity target (x) or any heuristic.
def build_training_target(x_tensor):
    # For simplicity: use identity target at LSTM output dimensionality by projecting input -> output
    # Here we just use zeros target with small magnitude so training doesn't blow up.
    # You can replace with a smarter label if you have one.
    batch = x_tensor.shape[0]
    seq = x_tensor.shape[1]
    return torch.zeros((batch, seq, OUTPUT_SIZE), device=x_tensor.device, dtype=torch.float32)

async def train_worker(save_every=50):
    step = 0
    while True:
        raw_data = await training_queue.get()
        if not raw_data or not raw_data.strip():
            continue
        try:
            # Preprocess
            x_tensor, rule_ids = preprocess_script(raw_data)
            x = x_tensor.to(device).unsqueeze(0).unsqueeze(0)  # (1,1,input_size)

            # Build a target y (self-supervised placeholder)
            y = build_training_target(x)

            # Train step
            optimizer.zero_grad()
            output, _ = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            step += 1
            if step % save_every == 0:
                torch.save(model.state_dict(), CACHE_PATH)
                print(f"💾 Trainer saved model to {CACHE_PATH} (step {step})")

            # optional debug
            print(f"🔁 Trained on cycle (rule_ids={rule_ids}) loss={loss.item():.6f}")

        except Exception as e:
            print("⚠️ Trainer error:", e)
            continue

async def main():
    # Run the trainer loop forever
    await train_worker()

if __name__ == "__main__":
    asyncio.run(main())
