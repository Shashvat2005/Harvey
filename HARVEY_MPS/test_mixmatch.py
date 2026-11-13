#!/usr/bin/env python3
"""Test MixMatch training path on MPS."""

import torch
import numpy as np
from models.target_models import ResNet18
from train_test import train_mixmatch_epoch
from torch.utils.data import DataLoader, TensorDataset

# Setup
device = torch.device('mps' if torch.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model
model = ResNet18(num_classes=10, fixed_model_parameter_seed=42).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Create dummy data loaders mimicking MixMatch structure
# Labeled data
n_labeled = 64
x_labeled = torch.randn(n_labeled, 3, 32, 32)
y_labeled = torch.randint(0, 10, (n_labeled,))
labeled_ds = TensorDataset(x_labeled, y_labeled)
labeled_loader = DataLoader(labeled_ds, batch_size=32, shuffle=True)

# Unlabeled data
n_unlabeled = 128
x_unlabeled = torch.randn(n_unlabeled, 3, 32, 32)
y_unlabeled = torch.randint(0, 10, (n_unlabeled,))  # dummy labels
unlabeled_ds = TensorDataset(x_unlabeled, y_unlabeled)
unlabeled_loader = DataLoader(unlabeled_ds, batch_size=32, shuffle=True)

dataloaders = {
    'train': labeled_loader,
    'train_unlabel': unlabeled_loader,
}

# MixMatch criterion
from utils.loss import MixMatchLoss
criterion = MixMatchLoss(rampup_length=100, lambda_u=15)

print("Testing train_mixmatch_epoch...")
try:
    loss = train_mixmatch_epoch(dataloaders, model, optimizer, criterion, step=0, device=device, NO_CLASSES=10)
    print(f"✓ MixMatch epoch completed successfully")
    print(f"  Loss: {loss:.4f}")
except Exception as e:
    print(f"✗ MixMatch epoch failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test with SSL disabled (normal training)
print("\nTesting normal training epoch (SSL disabled)...")
from train_test import train_epoch
import torch.nn as nn

model2 = ResNet18(num_classes=10, fixed_model_parameter_seed=43).to(device)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
criterion2 = nn.CrossEntropyLoss(reduction='none')

# Create simple dataloaders for normal training
train_ds = TensorDataset(torch.randn(64, 3, 32, 32), torch.randint(0, 10, (64,)))
test_ds = TensorDataset(torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,)))
test_bad_ds = TensorDataset(torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,)))

dataloaders2 = {
    'train': DataLoader(train_ds, batch_size=32, shuffle=True),
    'test': DataLoader(test_ds, batch_size=32),
    'test_bad': DataLoader(test_bad_ds, batch_size=32),
    'train_clean': DataLoader(train_ds, batch_size=32, shuffle=True),
}

try:
    # Need to create a mock cfgs object
    class MockConfig:
        LGA = False
        LGA_gamma = 0.01
    
    cfgs = MockConfig()
    
    loss, acc = train_epoch(
        {'backbone': model2}, 'SCE', criterion2, 
        {'backbone': optimizer2}, {}, dataloaders2, 
        epoch=0, epoch_loss=100, device=device, MARGIN=1.0, WEIGHT=1.0,
        log_data_map=False, cfgs=cfgs, warmup_train=False, VIS=True
    )
    print(f"✓ Normal training epoch completed")
    print(f"  Loss: {loss:.4f}, Acc: {acc:.2f}%")
except Exception as e:
    print(f"✗ Normal training epoch failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 50)
print("ALL MIXMATCH TESTS PASSED")
print("=" * 50)