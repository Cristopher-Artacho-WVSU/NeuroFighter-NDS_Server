# preprocessing.py
import json
import numpy as np
import torch

# Keep a small cache for last cycles if you want (not used by LSTM input directly now)
past_cycles = []

# Constants describing expected input layout
RULES_PER_SCRIPT = 5
FEATURES_PER_RULE = 3  # weight, was_used, distance
PLAYER_METRICS_COUNT = 4  # upper_taken, lower_taken, upper_landed, lower_landed

def preprocess_script(raw_data: str):
    """
    Parse the raw JSON string that Godot sends and convert it to a fixed-length
    torch tensor suitable for the LSTM model. Also return the ordered rule_ids list.

    Returns:
        (tensor, rule_ids)
        - tensor: torch.float32 of shape (FEATURE_VECTOR_LENGTH,)
        - rule_ids: list[int] of length RULES_PER_SCRIPT
    """
    if not raw_data or not raw_data.strip():
        raise ValueError("Empty payload")

    data = json.loads(raw_data)

    rule_ids = []

    # Collect up to RULES_PER_SCRIPT rules (maintain order)
    features = []
    for rule in data.get("script", [])[:5]:
        # Extract rule ID from either key
        rule_id = rule.get("rule_id")
        if rule_id is None:
            rule_id = rule.get("ruleID", -1)
        rule_ids.append(rule_id)

        features.append([
            float(rule.get("weight", 0.0)),
            float(rule.get("was_used", False)),
            float(rule.get("distance", 0.0))
        ])



    # Pad missing rules with zeros and -1 rule_id
    while len(features) < RULES_PER_SCRIPT:
        features.append([0.0, 0.0, 0.0])
        rule_ids.append(-1)

    features_arr = np.array(features, dtype=np.float32).reshape(RULES_PER_SCRIPT * FEATURES_PER_RULE)

    # Player metrics (always 4 values, normalized)
    params = data.get("parameters", {})
    player_metrics = np.array([
        float(params.get("upper_attacks_taken", 0.0)),
        float(params.get("lower_attacks_taken", 0.0)),
        float(params.get("upper_attacks_landed", 0.0)),
        float(params.get("lower_attacks_landed", 0.0))
    ], dtype=np.float32)

    total = np.sum(player_metrics) + 1e-6
    player_metrics = player_metrics / total

    # Previous fitness
    prev_fitness = np.array([float(data.get("fitness", 0.0))], dtype=np.float32)

    # Final input vector length = RULES_PER_SCRIPT*FEATURES_PER_RULE + PLAYER_METRICS_COUNT + 1
    final_input = np.concatenate([features_arr, player_metrics, prev_fitness]).astype(np.float32)

    # Keep last cycles if needed for debugging/analysis
    past_cycles.append(final_input)
    if len(past_cycles) > 10:
        past_cycles.pop(0)

    return torch.tensor(final_input, dtype=torch.float32), rule_ids
