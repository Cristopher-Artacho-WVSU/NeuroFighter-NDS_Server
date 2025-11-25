# lstm_server.py
import asyncio
import json
import os
import torch
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
from preprocessing import preprocess_script
from lstm_model import LSTMNet
from trainer_queue import training_queue
from fastapi.responses import HTMLResponse
from collections import Counter
from fastapi.responses import JSONResponse

app = FastAPI()
script_queue = asyncio.Queue()


# MODEL CONFIG (must match preprocess final vector length)
INPUT_SIZE = 20   # RULES_PER_SCRIPT*3 (15) + 4 player metrics + 1 fitness = 20
HIDDEN_SIZE = 64
OUTPUT_SIZE = 5   # one output per rule in the script
history_table = []
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

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            # put raw text into a local queue for worker to process
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
                rule_id = int(rule_ids[i]) if i < len(rule_ids) else -1
                weight_adjustment = float(pred_vector[i])
                
                # Get old weight from input
                old_weight = 0.0
                try:
                    old_weight = float(json.loads(raw_data)["script"][i].get("weight", 0.0))
                except (IndexError, KeyError, TypeError):
                    old_weight = 0.0

                new_weight = old_weight + weight_adjustment

                # Append to formatted response
                formatted.append({
                    "rule_id": rule_id,
                    "weight_adjustment": weight_adjustment
                })

                # Append to history table
                try:
                    cycle_id = int(json.loads(raw_data).get("cycle_id", -1))
                except Exception:
                    cycle_id = -1

                history_table.append({
                    "cycle_id": cycle_id,
                    "rule_id": rule_id,
                    "old_weight": old_weight,
                    "new_weight": new_weight,
                    "weight_adjustment": weight_adjustment
                })

            # Send response
            await websocket.send_text(json.dumps(formatted))

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

@app.get("/history", response_class=HTMLResponse)
async def get_history():
    # Build HTML table with auto-refresh every 2 seconds
    html = """
    <html>
    <head>
        <title>LSTM History Table</title>
        <style>
            table { border-collapse: collapse; width: 80%; margin: 20px auto; }
            th, td { border: 1px solid #333; padding: 8px 12px; text-align: center; }
            th { background-color: #555; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
        </style>
        <script>
            setTimeout(function(){
                window.location.reload(1);
            }, 2000); // refresh every 2 seconds
        </script>
    </head>
    <body>
        <h2 style="text-align:center;">LSTM Rule History Table</h2>
        <table>
            <tr>
                <th>Cycle ID</th>
                <th>Rule ID</th>
                <th>Old Weight</th>
                <th>New Weight</th>
                <th>Weight Adjustment</th>
            </tr>
    """

    for entry in history_table:
        html += f"""
            <tr>
                <td>{entry['cycle_id']}</td>
                <td>{entry['rule_id']}</td>
                <td>{entry['old_weight']:.4f}</td>
                <td>{entry['new_weight']:.4f}</td>
                <td>{entry['weight_adjustment']:.4f}</td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


def get_rule_counts():
    """
    Returns a dict mapping rule_id -> number of times it appeared in scripts
    """
    counts = Counter()
    for entry in history_table:
        rule_id = entry["rule_id"]
        if rule_id != -1:
            counts[rule_id] += 1
    return counts



@app.get("/bar", response_class=HTMLResponse)
async def bar_chart():
    counts = get_rule_counts()
    labels = list(counts.keys())
    values = list(counts.values())

    html = f"""
    <html>
    <head>
        <title>Rule Occurrences Bar Chart</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ text-align:center; font-family: Arial, sans-serif; }}
            .chart-container {{
                width: 1600px;   /* double original width */
                height: 800px;   /* double original height */
                margin: 0 auto;
            }}
        </style>
    </head>
    <body>
        <h2>Rule Occurrences in Scripts</h2>
        <div class="chart-container">
            <canvas id="ruleChart"></canvas>
        </div>
        <script>
            const ctx = document.getElementById('ruleChart').getContext('2d');
            const data = {{
                labels: {labels},
                datasets: [{{
                    label: 'Occurrences',
                    data: {values},
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }}]
            }};
            const config = {{
                type: 'bar',
                data: data,
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            precision: 0
                        }}
                    }}
                }}
            }};
            const ruleChart = new Chart(ctx, config);

            // Auto-refresh chart every 2 seconds
            setInterval(async () => {{
                const response = await fetch('/bar-data');
                const newData = await response.json();
                ruleChart.data.labels = newData.labels;
                ruleChart.data.datasets[0].data = newData.values;
                ruleChart.update();
            }}, 2000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/bar-data")
async def bar_data():
    counts = get_rule_counts()
    return JSONResponse({"labels": list(counts.keys()), "values": list(counts.values())})
