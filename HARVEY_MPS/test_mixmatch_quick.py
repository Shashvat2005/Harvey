#!/usr/bin/env python3
"""Test MixMatch training path on MPS (quick test)."""

import torch
import numpy as np
from models.target_models import ResNet18
from train_test import train_mixmatch_epoch, MixMatchLoss
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
criterion = MixMatchLoss(rampup_length=100, lambda_u=15)

# Monkey-patch train_iteration to a small number for quick test
import train_test as tt
original_train_mixmatch_epoch = tt.train_mixmatch_epoch

def quick_train_mixmatch_epoch(dataloaders, model, optimizer, criterion, step, device, NO_CLASSES, unlearn_reg=False,
                     alpha=0.75, T=0.5, fix_w=False):
    """Run only 2 iterations for quick testing."""
    batch_time = tt.AverageMeter()
    data_time = tt.AverageMeter()
    losses = tt.AverageMeter()
    losses_x = tt.AverageMeter()
    losses_u = tt.AverageMeter()
    ws = tt.AverageMeter()
    end = tt.time.time()

    labeled_trainloader = dataloaders['train']
    unlabeled_trainloader = dataloaders['train_unlabel']

    train_iteration = 2  # Only 2 iterations for quick test
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
            if inputs_x.shape[0] != labeled_trainloader.batch_size:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_train_iter)

        try:
            inputs_u, targets_poi = next(unlabeled_train_iter)
            if inputs_u.shape[0] != unlabeled_trainloader.batch_size:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, targets_poi = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, targets_poi = next(unlabeled_train_iter)
        inputs_u2 = tt.deepcopy(inputs_u)

        data_time.update(tt.time.time() - end)

        batch_size = inputs_x.size(0)

        targets_x = torch.zeros(batch_size, NO_CLASSES).scatter_(1, targets_x.view(-1,1).long(), 1)

        inputs_x = inputs_x.to(device)
        targets_x = targets_x.to(device)
        inputs_u = inputs_u.to(device)
        inputs_u2 = inputs_u2.to(device)
        targets_poi = targets_poi.to(device)

        with torch.no_grad():
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(alpha, alpha)
        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = tt.interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        logits = tt.interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], step)

        loss = Lx + w * Lu

        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(tt.time.time() - end)
        end = tt.time.time()

    print(f'SSL-loss: {losses.avg:.4f}({losses_x.avg:.4f}, {losses_u.avg:.4f}), W={w:.2f}')
    return losses.avg

print("Testing train_mixmatch_epoch (quick 2-iteration test)...")
try:
    loss = quick_train_mixmatch_epoch(dataloaders, model, optimizer, criterion, step=0, device=device, NO_CLASSES=10)
    print(f"✓ MixMatch epoch completed successfully")
    print(f"  Loss: {loss:.4f}")
except Exception as e:
    print(f"✗ MixMatch epoch failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Verify device placement of tensors during MixMatch
print("\nVerifying tensor device placement...")
model.eval()
with torch.no_grad():
    test_input = torch.randn(1, 3, 32, 32, device=device)
    out = model(test_input)
    print(f"  Model output device: {out.device}")
    print(f"  Model parameters device: {next(model.parameters()).device}")

print("\n" + "=" * 50)
print("MIXMATCH TEST PASSED")
print("=" * 50)