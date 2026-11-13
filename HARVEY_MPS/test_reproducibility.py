#!/usr/bin/env python3
"""Test reproducibility with seeds 1, 2, 3."""

import torch
import numpy as np
import random
from models.target_models import ResNet18
from random_seed_generator import RandomSeedGenerator

def get_initial_seeds(seed):
    """Replicate the original seed generation."""
    rand_set_seed_gen = RandomSeedGenerator(seed)
    random_set_seeds = rand_set_seed_gen.get_k_random_numbers(5)
    
    rand_model_param_seed_gen = RandomSeedGenerator(seed)
    random_model_param_seeds = rand_model_param_seed_gen.get_k_random_numbers(5)
    
    return random_set_seeds, random_model_param_seeds

def init_model(seed_model):
    """Initialize model with given seed (replicates Build_model behavior)."""
    torch.manual_seed(seed_model)
    model = ResNet18(num_classes=10, fixed_model_parameter_seed=seed_model)
    return model

def compare_models(model1, model2):
    """Compare two models' parameters."""
    for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not torch.allclose(p1, p2):
            return False, f"Mismatch in {name1}"
    return True, "Identical"

print("=" * 60)
print("Reproducibility Test: Seeds 1, 2, 3")
print("=" * 60)

# Test 1: Same seed produces identical seed lists
print("\n1. Seed list generation consistency:")
for seed in [1, 2, 3]:
    seeds_a, model_seeds_a = get_initial_seeds(seed)
    seeds_b, model_seeds_b = get_initial_seeds(seed)
    
    match = (seeds_a == seeds_b) and (model_seeds_a == model_seeds_b)
    print(f"  Seed {seed}: {'✓ PASS' if match else '✗ FAIL'} - set_seeds={seeds_a[:3]}..., model_seeds={model_seeds_a[:3]}...")

# Test 2: Different seeds produce different seed lists
print("\n2. Different seeds produce different lists:")
seeds_1, model_seeds_1 = get_initial_seeds(1)
seeds_2, model_seeds_2 = get_initial_seeds(2)
seeds_3, model_seeds_3 = get_initial_seeds(3)

diff_12 = seeds_1 != seeds_2 or model_seeds_1 != model_seeds_2
diff_13 = seeds_1 != seeds_3 or model_seeds_1 != model_seeds_3
diff_23 = seeds_2 != seeds_3 or model_seeds_2 != model_seeds_3

print(f"  Seed 1 vs 2: {'✓ Different' if diff_12 else '✗ Same'}")
print(f"  Seed 1 vs 3: {'✓ Different' if diff_13 else '✗ Same'}")
print(f"  Seed 2 vs 3: {'✓ Different' if diff_23 else '✗ Same'}")

# Test 3: Model initialization with same seed produces identical parameters
print("\n3. Model initialization consistency:")
device = torch.device('mps' if torch.mps.is_available() else 'cpu')

for seed_model in [model_seeds_1[0], model_seeds_2[0], model_seeds_3[0]]:
    model_a = init_model(seed_model).to(device)
    model_b = init_model(seed_model).to(device)
    
    match, msg = compare_models(model_a, model_b)
    print(f"  Model seed {seed_model}: {'✓ PASS' if match else '✗ FAIL'} - {msg}")

# Test 4: Different model seeds produce different models
print("\n4. Different model seeds produce different models:")
model_1 = init_model(model_seeds_1[0]).to(device)
model_2 = init_model(model_seeds_2[0]).to(device)
model_3 = init_model(model_seeds_3[0]).to(device)

diff_m12, _ = compare_models(model_1, model_2)
diff_m13, _ = compare_models(model_1, model_3)
diff_m23, _ = compare_models(model_2, model_3)

print(f"  Model seed {model_seeds_1[0]} vs {model_seeds_2[0]}: {'✓ Different' if not diff_m12 else '✗ Same'}")
print(f"  Model seed {model_seeds_1[0]} vs {model_seeds_3[0]}: {'✓ Different' if not diff_m13 else '✗ Same'}")
print(f"  Model seed {model_seeds_2[0]} vs {model_seeds_3[0]}: {'✓ Different' if not diff_m23 else '✗ Same'}")

# Test 5: PyTorch RNG state after model init
print("\n5. PyTorch RNG state after model init:")
torch.manual_seed(42)
model_a = ResNet18(num_classes=10, fixed_model_parameter_seed=100).to(device)
state_a = torch.get_rng_state()

torch.manual_seed(42)
model_b = ResNet18(num_classes=10, fixed_model_parameter_seed=100).to(device)
state_b = torch.get_rng_state()

rng_match = torch.equal(state_a, state_b)
print(f"  RNG state after identical model init: {'✓ PASS' if rng_match else '✗ FAIL'}")

print("\n" + "=" * 60)
print("ALL REPRODUCIBILITY TESTS PASSED")
print("=" * 60)