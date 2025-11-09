import json
import numpy as np
import torch


# Keep a small cache for past cycles
past_cycles = []

def preprocess_script(raw_data: str):
    data = json.loads(raw_data)
    
    # Example: features extraction
    features = []
    for rule in data["rules"]:
        features.append([
            rule.get("weight", 0.0),
            int(rule.get("was_used", False)),
            rule.get("distance", 0.0),
            # other numeric features
        ])
    
    # Include past cycles
    if past_cycles:
        past_features = np.mean(past_cycles[-3:], axis=0)  # last 3 cycles
        features = np.concatenate([features, past_features], axis=0)
    
    # Normalize player metrics
    player_metrics = np.array([
        data["parameters"]["upper_attacks_taken"],
        data["parameters"]["lower_attacks_taken"],
        data["parameters"]["upper_attacks_landed"],
        data["parameters"]["lower_attacks_landed"]
    ])
    player_metrics = player_metrics / (player_metrics.sum() + 1e-6)
    
    # Previous fitness
    prev_fitness = data.get("fitness", 0.0)
    
    # Combine all features
    final_input = np.concatenate([features.flatten(), player_metrics, [prev_fitness]])
    
    past_cycles.append(final_input)
    return torch.tensor(final_input, dtype=torch.float32)
