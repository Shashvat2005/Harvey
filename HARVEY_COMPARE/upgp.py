"""UPGP Eqs. 17-19: CPU SVD of stacked layer gradient rows.

HARVEY supplies estimated sets instead of UPGP's perturbation detector.
Projection uses G @ (I - V diag(lambda) V.T), avoiding dense projectors.
"""
import torch
import torch.nn.functional as F


def basis_from_rows(rows, energy=0.97, alpha=10.):
    matrix = torch.cat(rows, dim=0).float().cpu()
    _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    total = singular.square().sum()
    if total <= 1e-20:
        return torch.empty(matrix.shape[1], 0), torch.empty(0)
    rank = min(len(singular), int(torch.searchsorted(singular.square().cumsum(0), energy * total)) + 1)
    weights = (alpha + 1) * singular[:rank] / (alpha * singular[:rank] + singular[0])
    return vh[:rank].T.contiguous(), weights


def project(gradient, basis):
    if gradient.ndim <= 1 or basis is None:
        return gradient
    vectors, weights = (v.to(gradient.device, gradient.dtype) for v in basis)
    flat = gradient.reshape(gradient.shape[0], -1)
    return (flat - ((flat @ vectors) * weights) @ vectors.T).reshape_as(gradient)


def build_bases(model, loader, device, batches=4, energy=0.97, alpha=10.):
    # Eval mode freezes BN statistics while differentiating clean loss.
    model.eval()
    rows = {n: [] for n, p in model.named_parameters() if p.ndim > 1}
    for index, (x, y) in enumerate(loader):
        if index >= batches:
            break
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x.to(device)), y.to(device)).backward()
        for n, p in model.named_parameters():
            if n in rows and p.grad is not None:
                rows[n].append(p.grad.detach().cpu().reshape(p.shape[0], -1).clone())
    model.zero_grad(set_to_none=True)
    if not all(rows.values()):
        raise ValueError('No gradients for UPGP projection; estimated benign set is empty.')
    return {n: basis_from_rows(r, energy, alpha) for n, r in rows.items()}


def unlearning_loss(model, benign, poison, device):
    x, y = benign
    xp, yp = poison
    normal = F.cross_entropy(model(x.to(device)), y.to(device))
    logits = model(xp.to(device))
    # Stable -log(1-p_target), computed through logsumexp over other classes.
    masked = logits.clone()
    masked.scatter_(1, yp.to(device)[:, None], -torch.inf)
    backdoor = (torch.logsumexp(logits, 1) - torch.logsumexp(masked, 1)).mean()
    return normal + backdoor
