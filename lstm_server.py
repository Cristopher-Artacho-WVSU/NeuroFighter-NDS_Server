# lstm_server.py
import asyncio
import json
import os
from typing import Literal, cast
import torch
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .preprocessing import preprocess_script
from .lstm_model import LSTMNet
from .trainer_queue import training_queue
from .analytics.main import analytics_router, rules_adjustments_history
from .analytics.data_cleanup import fix_json_format, data_processing_queue

app = FastAPI()
app.include_router(analytics_router)
app.mount("/static", StaticFiles(directory="analytics/static"), name="static")
script_queue = asyncio.Queue()

# MODEL CONFIG (must match preprocess final vector length)
INPUT_SIZE = 20   # RULES_PER_SCRIPT*3 (15) + 4 player metrics + 1 fitness = 20
HIDDEN_SIZE = 64
OUTPUT_SIZE = 5   # one output per rule in the script

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMNet(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(device)

# Try load cached model if exists
CACHE_PATH = "lstm_cached.pth"
if os.path.exists(CACHE_PATH):
    try:
        model.load_state_dict(torch.load(CACHE_PATH, map_location=device))
        print(f"✅ Loaded cached model from {CACHE_PATH}")
    except Exception as e:
        print("⚠️ Failed to load cached model:", e)

model.eval()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            # put raw text into a local queue for worker to process
            await data_processing_queue.put(text)
            await script_queue.put((text, websocket))
    except WebSocketDisconnect:
        print("Client disconnected")

async def lstm_worker():
    """
    Worker that processes incoming cycles (from script_queue), runs prediction,
    maps outputs to rule_ids, then replies through the websocket.
    It also forwards raw_data strings to training_queue for the trainer to consume.
    """
    while True:
        raw_data, websocket = await script_queue.get()
        print("🟢 Received raw_data:", raw_data)
        if not raw_data or not raw_data.strip():
            # ignore empty payloads
            continue

        try:
            # Preprocess -> returns tensor and rule_ids
            input_tensor, rule_ids = preprocess_script(raw_data)

            # guard lengths
            if len(rule_ids) < OUTPUT_SIZE:
                # pad rule_ids
                rule_ids = rule_ids + [-1] * (OUTPUT_SIZE - len(rule_ids))
            elif len(rule_ids) > OUTPUT_SIZE:
                rule_ids = rule_ids[:OUTPUT_SIZE]

            # Prepare input shape (batch=1, seq_len=1, features=INPUT_SIZE)
            input_tensor = input_tensor.to(device).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                output, _ = model(input_tensor)

            pred_vector = output.squeeze(0).squeeze(0).cpu().tolist()

            # Build structured response: [{rule_id, weight_adjustment}, ...]
            formatted = []
            for i in range(OUTPUT_SIZE):
                formatted.append({
                    "rule_id": int(rule_ids[i]) if i < len(rule_ids) else -1,
                    "weight_adjustment": float(pred_vector[i])
                })

            # Send response
            await websocket.send_text(json.dumps(formatted))

            # For analytics
            rules_adjustments_history.append(cast(list[dict[Literal["rule_id", "weight_adjustment"], int | float]], formatted))

            # Debug print
            print("✅ Sent prediction back to client:", formatted)

            # Also forward raw_data to trainer queue (non-blocking)
            try:
                # put_nowait to avoid blocking; trainer will process asynchronously
                training_queue.put_nowait(raw_data)
            except asyncio.QueueFull:
                # If queue is full, drop training sample (or handle differently)
                print("⚠️ Training queue full — dropped a sample")

        except Exception as e:
            print("⚠️ Error processing LSTM data:", e)

@app.on_event("startup")
async def startup_event():
    # start worker
    asyncio.create_task(lstm_worker())
    print("🔵 LSTM server started and worker created.")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")

