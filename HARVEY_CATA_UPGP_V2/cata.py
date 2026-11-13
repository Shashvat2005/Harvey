"""CATA equations 8-16, with an explicit HARVEY poison-learning application base.

Vectors always use one immutable anchor. Positive/negative sums and nonzero
counts are sufficient statistics for the full historical vector memory.
"""
from copy import deepcopy
import math

import torch
from torch import nn


def snapshot(model):
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


def topk_mask(vector, keep_ratio):
    """Global magnitude threshold (Eq. 9); threshold ties are retained."""
    if not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1:
        raise ValueError('CATA keep_ratio must be in (0, 1].')
    if vector.numel() == 0 or not torch.isfinite(vector).all():
        raise ValueError('CATA needs a nonempty, finite task vector.')
    count = max(1, math.ceil(vector.numel() * keep_ratio))
    threshold = vector.abs().kthvalue(vector.numel() - count + 1).values
    return vector * (vector.abs() >= threshold)


class ConflictAverseMemory:
    """Exact sign election and disjoint mean, including zero-vote ties."""

    def __init__(self, size):
        self.positive_sum = torch.zeros(size)
        self.negative_sum = torch.zeros(size)
        self.positive_count = torch.zeros(size, dtype=torch.int32)
        self.negative_count = torch.zeros(size, dtype=torch.int32)
        self.requests = 0

    def add(self, vector):
        vector = vector.detach().to(device='cpu', dtype=torch.float32)
        if vector.shape != self.positive_sum.shape or not torch.isfinite(vector).all():
            raise ValueError('CATA vector has an incompatible shape or nonfinite values.')
        self.positive_sum += vector.clamp_min(0)
        self.negative_sum += vector.clamp_max(0)
        self.positive_count += (vector > 0).to(torch.int32)
        self.negative_count += (vector < 0).to(torch.int32)
        self.requests += 1

    def aggregate(self):
        vote = self.positive_sum + self.negative_sum
        positive_mean = self.positive_sum / self.positive_count.clamp_min(1)
        negative_mean = self.negative_sum / self.negative_count.clamp_min(1)
        return torch.where(vote > 0, positive_mean,
                           torch.where(vote < 0, negative_mean, torch.zeros_like(vote)))


class CATAUnlearner:
    def __init__(self, model, keep_ratio=0.3, scale=0.7, epochs=4,
                 lr=1e-5, weight_decay=0.1):
        if not math.isfinite(keep_ratio) or not 0 < keep_ratio <= 1:
            raise ValueError('CATA keep_ratio must be in (0, 1].')
        if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
            raise ValueError('CATA epochs must be a positive integer.')
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError('CATA lr must be positive and finite.')
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('CATA scale must be nonnegative and finite.')
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError('CATA weight_decay must be nonnegative and finite.')
        self.anchor = snapshot(model)
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        if not self.names:
            raise ValueError('CATA requires trainable parameters.')
        self.anchor_vector = self._vector(model)
        self.memory = ConflictAverseMemory(self.anchor_vector.numel())
        self.learning_state = snapshot(model)
        self.keep_ratio, self.scale = keep_ratio, scale
        self.epochs, self.lr, self.weight_decay = epochs, lr, weight_decay

    def _vector(self, model):
        parameters = dict(model.named_parameters())
        return torch.cat([parameters[n].detach().cpu().float().reshape(-1)
                          for n in self.names])

    def restore_learning_state(self, model):
        """Remove the previous correction before HARVEY learns the next poison set."""
        model.load_state_dict(self.learning_state, strict=True)

    def apply(self, model, base_state=None):
        """Eq. 16 by default; HARVEY explicitly supplies its poison-trained base.

        Buffers (including BatchNorm statistics/counters) come from the base.
        The forget-only adaptation branch never changes the live model's buffers.
        """
        base = self.anchor if base_state is None else base_state
        model.load_state_dict(base, strict=True)
        correction = self.memory.aggregate()
        offset = 0
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name not in self.names:
                    continue
                count = parameter.numel()
                delta = correction[offset:offset + count].view(parameter.shape)
                parameter.add_(delta.to(parameter.device, parameter.dtype), alpha=self.scale)
                offset += count

    def unlearn(self, model, forget_loader):
        """Learn the forget set from the anchor, negate, trim, merge, and apply.

        Empty requests add no vector but still apply the existing correction to
        the newly learned poison model. No ground-truth poison IDs are used.
        """
        if [n for n, p in model.named_parameters() if p.requires_grad] != self.names:
            raise ValueError('CATA model parameters changed; initialize a new unlearner.')
        learning_state = snapshot(model)
        branch = deepcopy(model)
        branch.load_state_dict(self.anchor, strict=True)
        branch.train()
        optimizer = torch.optim.AdamW(
            (p for p in branch.parameters() if p.requires_grad),
            lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        device = next(branch.parameters()).device
        examples = 0
        final_loss = None
        for _ in range(self.epochs):
            for data in forget_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(branch(inputs), labels)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss during CATA forget-set adaptation.')
                loss.backward()
                optimizer.step()
                examples += labels.size(0)
                final_loss = loss.detach().item()
        if examples:
            task = topk_mask(self.anchor_vector - self._vector(branch), self.keep_ratio)
            self.memory.add(task)
            nonzero = int(torch.count_nonzero(task))
        else:
            nonzero = 0
        self.learning_state = learning_state
        self.apply(model, base_state=self.learning_state)
        return {'requests': self.memory.requests, 'examples_seen': examples,
                'task_nonzero': nonzero, 'parameters': self.anchor_vector.numel(),
                'adaptation_loss': final_loss}


def run_cata_unlearning(model, forget_loader, options, state):
    """Called only at HARVEY's Stage-2 unlearning site; state survives cycles."""
    if state.get('engine') is None:
        state['engine'] = CATAUnlearner(
            model, keep_ratio=options.KEEP_RATIO, scale=options.SCALE,
            epochs=options.EPOCHS, lr=options.LR, weight_decay=options.WEIGHT_DECAY)
    stats = state['engine'].unlearn(model, forget_loader)
    print('CATA: requests={requests}, examples_seen={examples_seen}, '
          'nonzero={task_nonzero}/{parameters}, adaptation_loss={adaptation_loss}'.format(**stats),
          flush=True)
    return stats
