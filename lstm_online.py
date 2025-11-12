import torch
from torch import nn, optim
from lstm_model import LSTMNet
from preprocessing import preprocess_script
import asyncio
import json
import numpy as np

# =====================================================
# ✅ CONFIGURATION
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Match this with the output of preprocess_script()

model = LSTMNet(input_size=20, hidden_size=64, output_size=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

training_queue = asyncio.Queue()

# Keep track of global stats
rule_usage_counter = {}     # {rule_id: usage_count}
rule_fitness_history = {}   # {rule_id: [fitness_values]}
last_hidden = None          # Persist hidden states between cycles


# =====================================================
# 📊 INTERPRETATION LOGIC
# =====================================================
def analyze_rules(script, executed_rules, fitness):
    for rule in script:
        rule_id = rule["rule_id"]
        used = rule["was_used"]
        weight = rule["weight"]

        if rule_id not in rule_usage_counter:
            rule_usage_counter[rule_id] = 0
            rule_fitness_history[rule_id] = []

        if used or rule_id in executed_rules:
            rule_usage_counter[rule_id] += 1
            rule_fitness_history[rule_id].append(fitness)
        else:
            # Even unused rules get tracked
            rule_fitness_history[rule_id].append(fitness * 0.5)


def generate_rule_recommendations():
    recommendations = []
    for rule_id, usage_count in rule_usage_counter.items():
        history = rule_fitness_history.get(rule_id, [])
        avg_fitness = np.mean(history) if history else 0
        effective = usage_count > 2 and avg_fitness > 0.5

        if effective:
            adjustment = min(0.05 + avg_fitness * 0.1, 0.2)
        else:
            adjustment = -min(0.05 + (0.5 - avg_fitness) * 0.1, 0.2)

        recommendations.append({
            "rule_id": rule_id,
            "avg_fitness": round(avg_fitness, 3),
            "usage_count": usage_count,
            "adjustment": round(adjustment, 3)
        })
    return recommendations


# =====================================================
# 🧩 TRAINING LOOP
# =====================================================
async def train_worker():
    global last_hidden, model

    while True:
        raw_data = await training_queue.get()
        try:
            # Convert incoming JSON to features
            x = preprocess_script(raw_data).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,input_size)
            data_dict = json.loads(raw_data)

            # Analyze rules for heuristic-based recommendation
            analyze_rules(data_dict["script"], data_dict["executed_rules"], data_dict["fitness"])

            # Prepare dummy target — self-supervised fine-tuning
            y = x.detach()

            model.train()
            optimizer.zero_grad()

            output, last_hidden = model(x, last_hidden)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            # Interpret recommendations
            recs = generate_rule_recommendations()

            print(f"\n✅ Processed cycle {data_dict['cycle_id']} | Fitness={data_dict['fitness']:.3f}")
            print(f"➡️ Final feature vector length: {x.numel()}")
            print(f"📊 Rule Recommendations:")
            for r in recs:
                direction = "⬆️" if r["adjustment"] > 0 else "⬇️"
                print(f"  Rule {r['rule_id']} → {direction} {r['adjustment']} (avg_fitness={r['avg_fitness']}, used={r['usage_count']})")

        except Exception as e:
            print("⚠️ Error processing LSTM data:", str(e))
            continue
