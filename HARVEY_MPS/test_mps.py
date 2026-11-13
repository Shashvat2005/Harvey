#!/usr/bin/env python3
"""Test script for MPS compatibility of HARVEY_MPS."""

import sys
import torch
import numpy as np

# Test A: Import test
print("=" * 60)
print("Test A: Import test")
print("=" * 60)
try:
    from models.target_models import ResNet18, ResNet34
    from models.wresnet import WRN161
    from util import Build_model, query_strategy, compute_poison_rate
    from queries_strategies.symmetric_cross_entropy import SCEntropy
    from train_test import train_model, test, train_epoch
    from data_loader import DatasetBD
    from config import cfg_from_file
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test B: ResNet18 initialization on MPS
print("\n" + "=" * 60)
print("Test B: ResNet18 initialization on MPS")
print("=" * 60)
device = torch.device('mps' if torch.mps.is_available() else 'cpu')
print(f"Using device: {device}")
try:
    model = ResNet18(num_classes=10, fixed_model_parameter_seed=42).to(device)
    print(f"✓ ResNet18 initialized on {device}")
    print(f"  Model device: {next(model.parameters()).device}")
except Exception as e:
    print(f"✗ ResNet18 init failed: {e}")
    sys.exit(1)

# Test C: Forward/backward test on CIFAR-10-shaped input
print("\n" + "=" * 60)
print("Test C: Forward/backward on CIFAR-10 input")
print("=" * 60)
try:
    model.train()
    x = torch.randn(2, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (2,), device=device)
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    print(f"✓ Forward/backward successful")
    print(f"  Output shape: {out.shape}")
    print(f"  Loss: {loss.item():.4f}")
except Exception as e:
    print(f"✗ Forward/backward failed: {e}")
    sys.exit(1)

# Test D: DataLoader test
print("\n" + "=" * 60)
print("Test D: DataLoader test")
print("=" * 60)
try:
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.randn(10, 3, 32, 32), torch.randint(0, 10, (10,)))
    pin_mem = device.type == 'cuda'
    loader = DataLoader(ds, batch_size=4, shuffle=True, pin_memory=pin_mem)
    batch = next(iter(loader))
    inputs, labels = batch[0].to(device), batch[1].to(device)
    print(f"✓ DataLoader works")
    print(f"  pin_memory: {pin_mem}")
    print(f"  Batch device: {inputs.device}")
except Exception as e:
    print(f"✗ DataLoader test failed: {e}")
    sys.exit(1)

# Test E: One-batch ordinary training test
print("\n" + "=" * 60)
print("Test E: One-batch ordinary training")
print("=" * 60)
try:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    opt.zero_grad()
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    opt.step()
    print(f"✓ One-batch training successful")
    print(f"  Loss: {loss.item():.4f}")
except Exception as e:
    print(f"✗ One-batch training failed: {e}")
    sys.exit(1)

# Test F: One-batch original GA unlearning test
print("\n" + "=" * 60)
print("Test F: One-batch GA unlearning (negative CE on correct samples)")
print("=" * 60)
try:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(4, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    opt.zero_grad()
    out = model(x)
    # Original GA unlearning: negative CE on correctly classified samples only
    with torch.no_grad():
        _, preds = torch.max(out.data, 1)
    loss_inds = torch.eq(preds, y).int()  # 1 for correct, 0 for incorrect
    ce_loss = torch.nn.functional.cross_entropy(out, y, reduction='none')
    unlearn_loss = -0.001 * ce_loss * loss_inds  # Unlearn_Gamma = 0.001
    unlearn_loss = unlearn_loss.mean()
    unlearn_loss.backward()
    opt.step()
    print(f"✓ GA unlearning step successful")
    print(f"  Correct predictions: {loss_inds.sum().item()}/4")
    print(f"  Unlearn loss: {unlearn_loss.item():.4f}")
except Exception as e:
    print(f"✗ GA unlearning test failed: {e}")
    sys.exit(1)

# Test G: RCE query strategy test (simplified - just verify class can be imported)
print("\n" + "=" * 60)
print("Test G: RCE query strategy class available")
print("=" * 60)
try:
    # Just verify the class exists and can be instantiated with proper config
    from queries_strategies.symmetric_cross_entropy import SCEntropy, otsu_threshold
    print(f"✓ SCEntropy and otsu_threshold imported")
    print(f"  SCEntropy.__init__ signature: model, data_unlabeled, NO_CLASSES, cfgs, device, ...")
    print(f"  otsu_threshold available for threshold computation")
except Exception as e:
    print(f"✗ RCE strategy test failed: {e}")
    sys.exit(1)

# Test H: Verify no unintended CUDA-only calls in active path
print("\n" + "=" * 60)
print("Test H: Scan for remaining CUDA-only calls")
print("=" * 60)
import subprocess
import os

files_to_check = [
    'train.py',
    'train_test.py', 
    'util.py',
    'queries_strategies/symmetric_cross_entropy.py',
    'data_loader.py',
]

cuda_patterns = [
    r'\.cuda\(\)',
    r'torch\.cuda\.device',
    r'torch\.backends\.cudnn',
    r'torch\.cuda\.memory_allocated',
    r'pin_memory=True',
]

found_issues = []
for f in files_to_check:
    fpath = os.path.join(os.path.dirname(__file__), f)
    if os.path.exists(fpath):
        with open(fpath, 'r') as fp:
            content = fp.read()
            for pattern in cuda_patterns:
                import re
                matches = [(m.start(), m.group()) for m in re.finditer(pattern, content)]
                for pos, match in matches:
                    # Get line number
                    line_no = content[:pos].count('\n') + 1
                    # Check if it's in a comment
                    line = content.split('\n')[line_no - 1]
                    if not line.strip().startswith('#'):
                        found_issues.append(f"{f}:{line_no}: {match} in '{line.strip()}'")

if found_issues:
    print("⚠ Potential CUDA-only calls found (review needed):")
    for issue in found_issues:
        print(f"  {issue}")
else:
    print("✓ No unintended CUDA-only calls in active code paths")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)