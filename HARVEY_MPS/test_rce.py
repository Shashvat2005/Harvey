#!/usr/bin/env python3
"""Test RCE query strategy on MPS."""

import torch
import numpy as np
from models.target_models import ResNet18
from queries_strategies.symmetric_cross_entropy import SCEntropy, otsu_threshold
from torch.utils.data import DataLoader, TensorDataset

# Setup
device = torch.device('mps' if torch.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model
model = ResNet18(num_classes=10, fixed_model_parameter_seed=42).to(device)
model.eval()

# Create dummy dataset with known poison/clean split
class DummyDataset:
    def __init__(self, n=200):
        self.n = n
        self.data = list(range(n))
        # Mark every 7th sample as poisoned (~14%)
        self.poison_ids = np.array([i for i in range(n) if i % 7 == 0])
        self.clean_ids = np.array([i for i in range(n) if i % 7 != 0])
        self.noise_ids = np.array([])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()

dummy_data = DummyDataset(200)
print(f"Dataset: {len(dummy_data)} samples, {len(dummy_data.poison_ids)} poisoned, {len(dummy_data.clean_ids)} clean")

# Create a minimal config object
class MockConfig:
    plot_hist = False
    BACKDOOR = type('obj', (object,), {'target_label': 0})()
    BATCH = 32
    SSL = type('obj', (object,), {'sl_train': False})()
    ref_stg = 'RCE'
    OPTIMIZER_BACKBONE = type('obj', (object,), {'NAME_REF': 'SGD'})()
    INIT_BUDGET = 100
    BUDGET = 100

cfgs = MockConfig()

# Test SCEntropy with RCE (alpha=0, beta=1)
print("\nTesting RCE query strategy (alpha=0, beta=1)...")
strategy = SCEntropy(
    {'backbone': model}, dummy_data, 10, cfgs, device,
    alpha=0.0, beta=1.0
)
strategy.subset = list(range(200))
strategy.cycle = 1
strategy.BATCH = 32

# Test get_predict_prob (this exercises the device transfer code)
print("  Running get_predict_prob...")
try:
    pin_mem = device.type == 'cuda'
    unlabeled_loader = DataLoader(dummy_data, batch_size=strategy.BATCH,
                                sampler=torch.utils.data.SequentialSampler(strategy.subset),
                                pin_memory=pin_mem)
    losses, labels = strategy.get_predict_prob(unlabeled_loader)
    U = losses.cpu().numpy()
    print(f"  ✓ get_predict_prob completed")
    print(f"    Loss range: [{U.min():.4f}, {U.max():.4f}]")
    print(f"    Loss mean: {U.mean():.4f}")
except Exception as e:
    print(f"  ✗ get_predict_prob failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test query method (full RCE splitting)
print("\n  Running query (RCE splitting)...")
try:
    arg, num_clean, best_th = strategy.query()
    print(f"  ✓ Query completed")
    print(f"    best_th: {best_th:.4f}")
    print(f"    num_clean: {num_clean}")
    print(f"    arg (sorted indices): {arg[:10]}...")
    
    # Verify threshold logic
    # For RCE (alpha=0), best_th should be 4.6052
    expected_th = 4.6052
    if abs(best_th - expected_th) < 0.01:
        print(f"    ✓ Threshold matches expected RCE value (4.6052)")
    else:
        print(f"    ⚠ Threshold differs from expected: got {best_th:.4f}, expected {expected_th}")
    
    # Verify split: low loss = poisoned, high loss = clean
    # arg is sorted by loss ascending
    # First half should be mostly poisoned (low RCE loss for correctly classified target)
    subset_size = 50
    subset = [strategy.subset[i] for i in arg[:subset_size]]
    clean_pool = [strategy.subset[i] for i in arg[-subset_size:]]
    
    poison_in_subset = sum(1 for i in subset if i in dummy_data.poison_ids)
    poison_in_clean = sum(1 for i in clean_pool if i in dummy_data.poison_ids)
    
    print(f"    Subset (low loss): {poison_in_subset}/{subset_size} poisoned")
    print(f"    Clean pool (high loss): {poison_in_clean}/{subset_size} poisoned")
    
except Exception as e:
    print(f"  ✗ Query failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test Otsu threshold function
print("\nTesting Otsu threshold function...")
test_losses = np.random.exponential(scale=2.0, size=1000)
th = otsu_threshold(test_losses, lbound=1.0, ubound=90.0, max_loss=15.0, norm_loss=False)
print(f"  ✓ Otsu threshold computed: {th:.4f}")

print("\n" + "=" * 50)
print("ALL RCE TESTS PASSED")
print("=" * 50)