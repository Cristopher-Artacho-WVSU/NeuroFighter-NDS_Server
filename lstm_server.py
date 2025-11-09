# lstm_server.py
import asyncio
import json
import torch
from torch import nn
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
from preprocessing import preprocess_script
from lstm_model import LSTMNet  # your model definition

app = FastAPI()
script_queue = asyncio.Queue()

# Load model once (cached)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMNet(input_size=10, hidden_size=64, output_size=10).to(device)
model.load_state_dict(torch.load("lstm_cached.pth", map_location=device))
model.eval()  # inference mode

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
        # Preprocess data
        input_tensor = preprocess_script(data)
        input_tensor = input_tensor.to(device).unsqueeze(0)  # batch=1
        with torch.no_grad():
            output = model(input_tensor)
        # Send suggested rules back
        await websocket.send_text(json.dumps(output.tolist()))
