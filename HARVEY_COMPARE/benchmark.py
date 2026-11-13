#!/usr/bin/env python3
"""Measure real CIFAR-10 training throughput; does not save a trained model."""
import json
import time
import torch
from experiment import parse_args, make_data, new_model, loader, ROOT

args = parse_args()
if args.profile != 'full':
    raise SystemExit('Benchmark requires --profile full for real CIFAR-10.')
device = torch.device('mps' if torch.backends.mps.is_available() and args.device != 'cpu' else 'cpu')
torch.set_num_threads(args.threads)
train, split, clean, bad, poison, labels = make_data(args)
model = new_model(args.seed, device)
optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=1e-4)
iterator = iter(loader(train, range(len(train)), args.batch_size, True))

def sync():
    if device.type == 'mps':
        torch.mps.synchronize()

samples = []
for step in range(12):
    x, y = next(iterator)
    sync()
    start = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))
    loss.backward()
    optimizer.step()
    sync()
    if step >= 2:
        samples.append(time.monotonic() - start)
seconds = sum(samples) / len(samples)
report = {'device': str(device), 'batch': args.batch_size, 'seconds_per_training_batch': seconds,
          'full_50000_epoch_minutes_compute_only': seconds * ((50000 + args.batch_size - 1) // args.batch_size) / 60,
          'final_100_epoch_hours_compute_only': seconds * ((45000 + args.batch_size - 1) // args.batch_size) * 100 / 3600,
          'note': 'Excludes loading, evaluation, reference learning, CATA, UPGP SVD and checkpoints; not a total runtime estimate.'}
print(json.dumps(report, indent=2))
(ROOT / 'benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
