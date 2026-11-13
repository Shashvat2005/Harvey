#!/usr/bin/env python3
"""Kaggle CUDA runner for HARVEY + CATA + UPGP; same learning protocol as local experiments."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms as T

from vendor.models.target_models import ResNet18
from cata import CATAUnlearner, snapshot
from upgp import build_bases, project, unlearning_loss

ROOT = Path(__file__).resolve().parent
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def save(path, state, device):
    state['rng'] = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                    torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
    temp = path.with_suffix('.tmp')
    torch.save(cpu_tree(state), temp)
    temp.replace(path)


def restore_rng(state, device):
    a, b, c, d = state['rng']
    random.setstate(a)
    np.random.set_state(b)
    torch.set_rng_state(c)
    if d is not None and device.type == 'cuda':
        torch.cuda.set_rng_state(d, device)


class Images(Dataset):
    """One uint8 image store, on-demand trigger; no full dataset copies."""
    def __init__(self, images, labels, poison=(), train=False, attack=False):
        self.images, self.labels = images, np.asarray(labels)
        self.poison = set(map(int, poison))
        self.attack = attack
        self.transform = T.Compose(([T.RandomCrop(32, padding=4), T.RandomRotation(10),
                                     T.RandomHorizontalFlip()] if train else []) +
                                   [T.ToTensor(), T.Normalize(MEAN, STD)])
        self.pattern = np.array(Image.open(ROOT / 'vendor/cifar_badnets.png').convert('RGB'))
        self.mask = self.pattern != 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = int(self.labels[index])
        if self.attack or index in self.poison:
            image = image.copy()
            image[self.mask] = self.pattern[self.mask]
            label = 0
        return self.transform(Image.fromarray(image)), label


def loader(dataset, ids, batch, shuffle=False):
    return DataLoader(torch.utils.data.Subset(dataset, list(map(int, ids))), batch_size=batch,
                      shuffle=shuffle, num_workers=0, pin_memory=False)


def make_data(args):
    if args.profile == 'smoke':
        rng = np.random.default_rng(args.seed)
        x = rng.integers(0, 256, (80, 32, 32, 3), dtype=np.uint8)
        y = np.arange(80) % 10
        xt = rng.integers(0, 256, (40, 32, 32, 3), dtype=np.uint8)
        yt = np.arange(40) % 10
    else:
        # Read existing caches only. New downloads always stay in the new folder.
        external_root = os.environ.get('HARVEY_CIFAR10_ROOT')
        candidates = ([Path(external_root)] if external_root else []) + [ROOT / 'data', ROOT.parent / 'HARVEY/data/CIFAR10',
                      ROOT.parent / 'HARVEY_MPS/data/CIFAR10', ROOT.parent / 'HARVEY_MPS/data']
        data_root = next((p for p in candidates if (p / 'cifar-10-batches-py/data_batch_1').exists()), ROOT / 'data')
        train = CIFAR10(str(data_root), train=True, download=data_root == ROOT / 'data')
        test = CIFAR10(str(data_root), train=False, download=data_root == ROOT / 'data')
        x, y, xt, yt = train.data, np.array(train.targets), test.data, np.array(test.targets)
    poison = np.random.default_rng(args.seed).permutation(len(y))[:int(len(y) * .1)]
    train = Images(x, y, poison, train=True)
    split = Images(x, y, poison)
    observed = np.array(y).copy()
    observed[poison] = 0
    clean = Images(xt, yt)
    non_target = np.flatnonzero(yt != 0)
    bad = Images(xt[non_target], yt[non_target], attack=True)
    return train, split, clean, bad, poison, observed


def new_model(seed, device):
    # Same initialization for all variants and final models.
    cuda_devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        model = ResNet18(num_classes=10)
    return model.to(device)


def supervised(model, data, ids, args, device, optimizer, ascent=False):
    model.train()
    total, loss_sum = 0, 0.
    for x, y in loader(data, ids, args.batch_size, True):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        losses = F.cross_entropy(logits, y, reduction='none')
        if ascent:
            losses = -.01 * losses * (logits.detach().argmax(1) == y)
        else:
            # LGA is used for all reference learning, including warmup.
            losses = torch.sign(losses - .01) * losses
        loss = losses.mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite reference loss.')
        loss.backward()
        optimizer.step()
        total += len(y)
        loss_sum += float(loss.detach()) * len(y)
    if not total:
        raise RuntimeError('Empty reference training subset.')
    return loss_sum / total


@torch.no_grad()
def rce(model, data, args, device):
    model.eval()
    losses = []
    for x, y in loader(data, range(len(data)), args.batch_size):
        probabilities = model(x.to(device)).softmax(1).clamp(1e-7, 1.)
        # Exact released RCELoss (one-hot target clamped to 1e-4).
        onehot = F.one_hot(y.to(device), 10).float().clamp(1e-4, 1.)
        losses.append(-(probabilities * onehot.log()).sum(1).cpu().numpy())
    return np.concatenate(losses)


def partition(losses, labels):
    order = np.argsort(losses)
    benign = np.flatnonzero(losses >= 4.6052)
    poison_count = len(losses) - len(benign)
    size = max(int(.5 * poison_count), int(.05 * len(losses)))
    subset = list(order[:size])
    counts = np.bincount(labels[subset], minlength=10)
    minimum = (np.bincount(labels, minlength=10) * .01).astype(int)
    for index in order[size:]:
        label = labels[index]
        if counts[label] < minimum[label]:
            subset.append(index)
            counts[label] += 1
    return np.array(subset), benign


@torch.no_grad()
def accuracy(model, data, args, device):
    model.eval()
    correct = total = 0
    for x, y in loader(data, range(len(data)), args.batch_size):
        pred = model(x.to(device)).argmax(1).cpu()
        correct += int((pred == y).sum())
        total += len(y)
    return 100 * correct / total


def metrics(model, clean, bad, args, device):
    return {'clean_accuracy': accuracy(model, clean, args, device),
            'asr': accuracy(model, bad, args, device)}


def log(run, event, **values):
    row = {'time': datetime.now().astimezone().isoformat(), 'event': event, **values}
    with (run / 'metrics.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device('cuda:0' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Select a GPU accelerator in Kaggle notebook settings.')
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Kaggle runner supports CUDA and CPU validation.')
    seed_all(args.seed)
    run_dir = ROOT / 'runs' / args.run_name / args.variant / f'seed{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {k: v for k, v in vars(args).items() if k != 'resume'}
    identity['resolved_device'] = str(device)
    identity['torch'] = torch.__version__
    identity['source_hashes'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(ROOT.rglob('*.py')) if 'runs' not in p.parts}
    metadata = run_dir / 'metadata.json'
    checkpoint = run_dir / 'checkpoint.pt'
    if metadata.exists():
        if not args.resume:
            raise RuntimeError(f'Run exists: {run_dir}. Use --resume or another --run-name.')
        if json.loads(metadata.read_text()) != identity:
            raise RuntimeError('Resume settings/code/environment differ from saved metadata.')
        if (run_dir / 'result.json').exists():
            print(f'Already complete: {run_dir}', flush=True)
            return
        if not checkpoint.exists():
            raise RuntimeError('No complete checkpoint; choose a new run name.')
    else:
        atomic_json(metadata, identity)
    train, split, clean, bad, poison_ids, observed = make_data(args)
    np.save(run_dir / 'poison_ids.npy', poison_ids)
    poison_hash = hashlib.sha256(poison_ids.tobytes()).hexdigest()
    model = new_model(args.seed, device)
    # Local checkpoints are trusted, never load checkpoints from unknown sources.
    state = torch.load(checkpoint, map_location='cpu', weights_only=False) if args.resume and checkpoint.exists() else {
        'phase': 'warmup', 'epoch': 0, 'cycle': 0, 'elapsed': 0., 'cata': None}
    if 'model' in state:
        model.load_state_dict(state['model'])
        restore_rng(state, device)
    started = time.monotonic()
    previous_elapsed = state['elapsed']

    def commit():
        state['model'] = snapshot(model)
        state['elapsed'] = previous_elapsed + time.monotonic() - started
        save(checkpoint, state, device)

    if state['phase'] == 'warmup':
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        if 'optimizer' in state:
            opt.load_state_dict(state['optimizer'])
        for epoch in range(state['epoch'], args.warmup_epochs):
            loss = supervised(model, split, range(len(split)), args, device, opt)
            state.update(epoch=epoch + 1, optimizer=opt.state_dict())
            log(run_dir, 'warmup', epoch=epoch + 1, loss=loss)
            commit()
        order = np.argsort(rce(model, split, args, device))
        half = int(.5 * len(split))
        state.update(phase='reference', epoch=0, cycle=1, subset=order[:half],
                     benign=order[half:], previous_clean=0)
        # HARVEY reinitializes the reference model for the first isolation cycle.
        model = new_model(args.seed, device)
        state.pop('optimizer', None)
        commit()

    if state['phase'] == 'reference':
        engine = state.get('cata')
        if engine is not None:
            state.pop('cata', None)
        while state['cycle'] <= args.cycles:
            cycle = state['cycle']
            if state['epoch'] == 0 and engine is not None:
                engine.restore_learning_state(model)
            opt = torch.optim.Adam(model.parameters(), lr=.001)
            if 'optimizer' in state:
                opt.load_state_dict(state['optimizer'])
            for epoch in range(state['epoch'], args.reference_epochs):
                loss = supervised(model, split, state['subset'], args, device, opt)
                state.update(epoch=epoch + 1, optimizer=opt.state_dict())
                log(run_dir, 'reference', cycle=cycle, epoch=epoch + 1, loss=loss)
                commit()
            # A checkpoint here contains the poison-trained model before unlearning.
            # Interrupted unlearning replays from that checkpoint with its saved RNG.
            if len(state['benign']):
                if args.variant == 'harvey':
                    supervised(model, split, state['benign'], args, device, opt, ascent=True)
                else:
                    if engine is None:
                        engine = CATAUnlearner(model, epochs=args.cata_epochs)
                    stats = engine.unlearn(model, loader(split, state['benign'], args.batch_size, True))
                    log(run_dir, 'cata', cycle=cycle, **stats)
            losses = rce(model, split, args, device)
            subset, benign = partition(losses, observed)
            if cycle == 1:
                state['first_reference'] = snapshot(model)
            log(run_dir, 'split', cycle=cycle, benign=len(benign), subset=len(subset))
            converged = len(benign) == state['previous_clean']
            state.update(subset=subset, benign=benign, previous_clean=len(benign),
                         losses=losses, cata=engine, cycle=cycle + 1, epoch=0)
            state.pop('optimizer', None)
            if converged or cycle == args.cycles:
                state['phase'] = 'meta'
            commit()
            if state['phase'] == 'meta':
                break
        del engine, opt

    if state['phase'] == 'meta':
        poison_pool = np.flatnonzero(state['losses'] < 4.6052)
        model.load_state_dict(state['first_reference'])
        first_losses = rce(model, split, args, device)
        recovered = poison_pool[first_losses[poison_pool] >= 4.6052]
        rest = poison_pool[first_losses[poison_pool] < 4.6052]
        main = np.concatenate([state['benign'], recovered])
        if not len(main):
            raise RuntimeError('Meta splitting produced no benign samples; inspect the reference logs.')
        np.savez(run_dir / 'split.npz', benign=main, suspected_poison=rest)
        # Ground truth is used only for auditing after the split is decided.
        detected = np.intersect1d(rest, poison_ids).size
        quality = {'benign_count': len(main), 'suspected_poison_count': len(rest),
                   'detection_precision': detected / len(rest) if len(rest) else None,
                   'detection_recall': detected / len(poison_ids),
                   'benign_contamination': np.intersect1d(main, poison_ids).size / len(main)}
        atomic_json(run_dir / 'split_quality.json', quality)
        log(run_dir, 'meta', **quality)
        # Fresh final model, identical initialization across variants.
        model = new_model(args.seed, device)
        state = {k: state[k] for k in ('elapsed',)} | {
            'phase': 'final', 'epoch': 0, 'main': main, 'rest': rest}
        commit()

    if state['phase'] == 'final':
        opt = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.milestones)
        if 'optimizer' in state:
            opt.load_state_dict(state['optimizer'])
            sched.load_state_dict(state['scheduler'])
        for epoch in range(state['epoch'], args.final_epochs):
            model.train()
            losses = []
            for x, y in loader(train, state['main'], args.batch_size, True):
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x.to(device)), y.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite final training loss.')
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            sched.step()
            values = metrics(model, clean, bad, args, device)
            log(run_dir, 'final', epoch=epoch + 1, loss=float(np.mean(losses)), **values)
            state.update(epoch=epoch + 1, optimizer=opt.state_dict(), scheduler=sched.state_dict())
            commit()
        torch.save(snapshot(model), run_dir / 'before_upgp.pt')
        state.update(phase='upgp', epoch=0, before_upgp=metrics(model, clean, bad, args, device))
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        commit()

    if state['phase'] == 'upgp':
        if args.variant == 'harvey_cata_upgp' and len(state['rest']):
            # Deterministic bounded sample of HARVEY-estimated benign training data.
            ids = np.random.default_rng(args.seed + 9000).permutation(state['main'])[:args.upgp_samples]
            if 'bases' not in state:
                state['bases'] = build_bases(model, loader(split, ids, args.batch_size), device,
                                             args.upgp_batches, args.upgp_energy, args.upgp_alpha)
                commit()
            opt = torch.optim.SGD(model.parameters(), lr=args.upgp_lr)  # no momentum/decay after projection
            for epoch in range(state['epoch'], args.upgp_epochs):
                model.eval()  # BN buffers fixed; gradients remain enabled
                normal_loader = loader(split, ids, args.batch_size, True)
                normal_iter = iter(normal_loader)
                for poison in loader(split, state['rest'], args.batch_size, True):
                    try:
                        benign = next(normal_iter)
                    except StopIteration:
                        normal_iter = iter(normal_loader)
                        benign = next(normal_iter)
                    opt.zero_grad(set_to_none=True)
                    loss = unlearning_loss(model, benign, poison, device)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite UPGP loss.')
                    loss.backward()
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad.copy_(project(param.grad, state['bases'].get(name)))
                    opt.step()
                log(run_dir, 'upgp', epoch=epoch + 1, **metrics(model, clean, bad, args, device))
                state['epoch'] = epoch + 1
                commit()
        elif args.variant == 'harvey_cata_upgp':
            log(run_dir, 'upgp_skipped', reason='No estimated poisoned samples')
        result = {**metrics(model, clean, bad, args, device), 'before_upgp': state['before_upgp'],
                  'variant': args.variant, 'seed': args.seed, 'profile': args.profile,
                  'protocol': args.protocol, 'poison_hash': poison_hash,
                  'elapsed_seconds': previous_elapsed + time.monotonic() - started,
                  'synthetic': args.profile == 'smoke', 'checkpoint_selection': 'last_epoch'}
        torch.save(snapshot(model), run_dir / 'final_model.pt')
        atomic_json(run_dir / 'result.json', result)
        log(run_dir, 'complete', **result)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['harvey', 'harvey_cata', 'harvey_cata_upgp'], required=True)
    p.add_argument('--profile', choices=['full', 'smoke'], default='full')
    p.add_argument('--protocol', choices=['release', 'paper-text'], default='release')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', choices=['auto', 'cuda:0', 'cpu'], default='auto')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--run-name', default='release-full')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--cata-epochs', type=int, default=4)
    p.add_argument('--upgp-epochs', type=int, default=5)
    p.add_argument('--upgp-lr', type=float, default=.001)
    p.add_argument('--upgp-samples', type=int, default=512)
    p.add_argument('--upgp-batches', type=int, default=4)
    p.add_argument('--upgp-energy', type=float, default=.97)
    p.add_argument('--upgp-alpha', type=float, default=10.)
    a = p.parse_args()
    if Path(a.run_name).name != a.run_name or a.run_name in ('.', '..'):
        p.error('--run-name must be one folder name')
    if a.batch_size < 2 or a.seed < 0 or a.threads < 1:
        p.error('batch size >= 2, seed >= 0, threads >= 1 required')
    if min(a.cata_epochs, a.upgp_epochs, a.upgp_samples, a.upgp_batches) < 1:
        p.error('epoch/sample/batch counts must be positive')
    if not 0 < a.upgp_energy <= 1 or not np.isfinite(a.upgp_alpha) or a.upgp_alpha < 0 or not np.isfinite(a.upgp_lr) or a.upgp_lr <= 0:
        p.error('invalid UPGP hyperparameters')
    a.warmup_epochs, a.reference_epochs, a.cycles, a.final_epochs = 20, (14 if a.protocol == 'release' else 10), 20, 100
    a.milestones = [60, 80, 90]
    if a.profile == 'smoke':
        a.warmup_epochs = a.reference_epochs = a.cycles = a.final_epochs = 1
        a.cata_epochs = a.upgp_epochs = 1
        a.batch_size = min(a.batch_size, 8)
        a.milestones = []
    return a


if __name__ == '__main__':
    run(parse_args())
