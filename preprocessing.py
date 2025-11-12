import json
import numpy as np
import torch

past_cycles = []

def preprocess_script(raw_data: str):
    data = json.loads(raw_data)

    # Expect exactly 5 rules
    features = []
    for rule in data.get("script", [])[:5]:  # Ensure max 5
        features.append([
            float(rule.get("weight", 0.0)),
            float(rule.get("was_used", False)),
            float(rule.get("distance", 0.0))
        ])

    # If fewer than 5 rules, pad to 5
    while len(features) < 5:
        features.append([0.0, 0.0, 0.0])

    features = np.array(features).flatten()  # shape = (15,)

    # Player metrics (always 4)
    player_metrics = np.array([
        data["parameters"].get("upper_attacks_taken", 0.0),
        data["parameters"].get("lower_attacks_taken", 0.0),
        data["parameters"].get("upper_attacks_landed", 0.0),
        data["parameters"].get("lower_attacks_landed", 0.0)
    ], dtype=np.float32)

    # Normalize metrics
    total = player_metrics.sum() + 1e-6
    player_metrics = player_metrics / total

    # Previous fitness
    prev_fitness = np.array([data.get("fitness", 0.0)], dtype=np.float32)

    # Combine all features
    final_input = np.concatenate([features, player_metrics, prev_fitness])  # shape = (20,)
    past_cycles.append(final_input)

    # Keep only last few for temporal context
    if len(past_cycles) > 5:
        past_cycles.pop(0)

    return torch.tensor(final_input, dtype=torch.float32)
