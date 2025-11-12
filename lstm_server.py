# lstm_server.py
import asyncio
import json
import torch
from torch import nn
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
from preprocessing import preprocess_script
from lstm_model import LSTMNet  # your model definition
import os
import asyncio

app = FastAPI()
script_queue = asyncio.Queue()

# Load model once (cached)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMNet(input_size=20, hidden_size=64, output_size=5).to(device)

torch.save(model.state_dict(), "lstm_cached.pth")
print("✅ Dummy LSTM model created as lstm_cached.pth")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await script_queue.put((data, websocket))
    except WebSocketDisconnect:
        print("Client disconnected")

async def lstm_worker():
    while True:
        data, websocket = await script_queue.get()
        try:
            # Preprocess incoming JSON
            input_tensor = preprocess_script(data)
            input_tensor = input_tensor.to(device).unsqueeze(0).unsqueeze(0)  # (batch, seq_len=1, features)
            
            # Run model (ignore hidden state)
            with torch.no_grad():
                output, _ = model(input_tensor)

            # Convert to plain Python list
            prediction = output.squeeze(0).squeeze(0).cpu().tolist()

            # Send back to client
            await websocket.send_text(json.dumps(prediction))
            print("✅ Sent prediction back to client:", prediction)

        except Exception as e:
            print("⚠️ Error processing LSTM data:", e)
            
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(lstm_worker())