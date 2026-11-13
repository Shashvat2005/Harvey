"""Anchored weight subtraction with training-only validation and raw-base fallback."""
from copy import deepcopy
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def make_groups(main, rest, first_rce, last_rce, labels, args):
    """No poison truth or test data enter group selection."""
    rng = np.random.default_rng(args.seed + 4100)
    main, rest, labels = np.asarray(main), np.asarray(rest), np.asarray(labels)
    high = rest[(first_rce[rest] <= args.poison_rce) &
                (last_rce[rest] <= args.poison_rce) & (labels[rest] == 0)]
    validation = []
    for cls in range(10):
        ids = rng.permutation(main[labels[main] == cls])
        # Retain at least one fitting example for a represented class.
        count = min(len(ids) - 1, max(1, int(round(len(ids) * args.clean_holdout)))) if len(ids) >= 2 else 0
        validation.extend(ids[:count])
    clean_val = np.array(validation, dtype=np.int64)
    clean_fit = np.setdiff1d(main, clean_val)
    high = rng.permutation(high)
    count = min(len(high) - 1, max(1, int(round(len(high) * args.poison_holdout)))) if len(high) >= 2 else 0
    poison_val, poison_fit = high[:count], high[count:]
    excluded = np.union1d(high, clean_val)
    base_train = np.setdiff1d(np.arange(len(labels)), excluded)
    groups = dict(base_train=base_train, likely_clean_fit=clean_fit,
                  likely_clean_validation=clean_val, suspicious_fit=poison_fit,
                  suspicious_validation=poison_val, high_confidence_poison=high)
    if not len(base_train):
        raise ValueError('Option B has an empty base training pool.')
    for holdout in (clean_val, poison_val):
        for fitting in (base_train, clean_fit, poison_fit):
            if np.intersect1d(holdout, fitting).size:
                raise ValueError('Validation samples overlap a fitting pool.')
    return groups


def task_delta(base, branch):
    parameters = dict(branch.named_parameters())
    return {name: parameters[name].detach().cpu() - base[name]
            for name in parameters if parameters[name].requires_grad}


def apply_delta(model, base, delta, alpha):
    """Every candidate starts from the same base; never subtract BN buffers."""
    model.load_state_dict(base, strict=True)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in delta:
                parameter.add_(delta[name].to(parameter.device, parameter.dtype), alpha=-alpha)


def recalibrate_bn(model, data, device):
    modules = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    momenta = [module.momentum for module in modules]
    model.eval()
    for module in modules:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    try:
        with torch.no_grad():
            for x, _ in data:
                model(x.to(device))
    finally:
        for module, momentum in zip(modules, momenta):
            module.momentum = momentum
        model.eval()


def fit_balanced(model, data, clean_ids, poison_ids, args, device, make_loader, *, stress=False):
    """Equal counts in every batch; BatchNorm buffers are frozen in adaptation."""
    rng = np.random.default_rng(args.seed + (4400 if stress else 4200))
    cap = args.stress_samples if stress else args.adapt_samples
    count = min(len(clean_ids), len(poison_ids), cap)
    if count < 1:
        raise ValueError('Balanced adaptation requires both training pools.')
    clean_ids = rng.permutation(clean_ids)[:count]
    poison_ids = rng.permutation(poison_ids)[:count]
    optimizer = torch.optim.SGD(model.parameters(), lr=args.stress_lr if stress else args.adapt_lr)
    model.eval()  # keeps BN buffers identical to the aligned base
    epochs = 1 if stress else args.adapt_epochs
    for _ in range(epochs):
        clean_ids, poison_ids = rng.permutation(clean_ids), rng.permutation(poison_ids)
        for benign, suspicious in zip(make_loader(data, clean_ids, args.batch_size // 2),
                                      make_loader(data, poison_ids, args.batch_size // 2)):
            x = torch.cat([benign[0], suspicious[0]]).to(device)
            y = torch.cat([benign[1], suspicious[1]]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite Option B adaptation loss.')
            loss.backward()
            optimizer.step()
    return count


@torch.no_grad()
def validation_metrics(model, data, groups, args, device, make_loader):
    model.eval()
    correct = total = target_correct = target_count = response = suspicious_count = 0
    confidence_sum = 0.
    for x, y in make_loader(data, groups['likely_clean_validation'], args.batch_size):
        pred = model(x.to(device)).argmax(1).cpu()
        correct += int((pred == y).sum())
        total += len(y)
        target_correct += int(((pred == y) & (y == 0)).sum())
        target_count += int((y == 0).sum())
    for x, _ in make_loader(data, groups['suspicious_validation'], args.batch_size):
        probabilities = model(x.to(device)).softmax(1)
        response += int((probabilities.argmax(1) == 0).sum())
        confidence_sum += float(probabilities[:, 0].sum())
        suspicious_count += len(x)
    return dict(clean_accuracy=100 * correct / total if total else None,
                target_accuracy=100 * target_correct / target_count if target_count else None,
                target_validation_count=target_count,
                suspicious_response=100 * response / suspicious_count if suspicious_count else None,
                suspicious_mean_target_probability=confidence_sum / suspicious_count if suspicious_count else None)


def choose_candidate(raw, calibrated, candidates, args):
    """Return a nonzero alpha only with a validation response improvement and guards."""
    required = ['clean_accuracy', 'target_accuracy', 'suspicious_response']
    if raw['target_validation_count'] < args.min_target_validation:
        return None, 'Insufficient likely-clean target-class validation examples'
    if any(raw[k] is None or calibrated[k] is None for k in required):
        return None, 'Missing validation measurements'
    safe = []
    for item in candidates:
        if item['alpha'] <= 0 or any(item[k] is None for k in required):
            continue
        if item['clean_accuracy'] < max(raw['clean_accuracy'], calibrated['clean_accuracy']) - args.max_acc_drop - 1e-8:
            continue
        if item['target_accuracy'] < max(raw['target_accuracy'], calibrated['target_accuracy']) - args.max_target_drop - 1e-8:
            continue
        improvement = min(raw['suspicious_response'], calibrated['suspicious_response']) - item['suspicious_response']
        if improvement < max(args.min_response_drop, 1e-8):
            continue
        safe.append(item)
    if not safe:
        return None, 'No nonzero candidate improved suspicious response within accuracy limits'
    return min(safe, key=lambda item: (item['suspicious_response'],
                                      item['suspicious_mean_target_probability'], item['alpha'])), 'Validation-selected subtraction'


@torch.no_grad()
def test_target_accuracy(model, clean, args, device, make_loader):
    model.eval()
    correct = count = 0
    for x, y in make_loader(clean, np.flatnonzero(clean.labels == 0), args.batch_size):
        pred = model(x.to(device)).argmax(1).cpu()
        correct += int((pred == y).sum())
        count += len(y)
    return 100 * correct / count


def run_option_b(model, data, clean, bad, groups, args, device, run_dir, make_loader, test_metrics, log):
    base = snapshot(model)
    raw_validation = validation_metrics(model, data, groups, args, device, make_loader)
    reason = 'Insufficient high-confidence suspicious fitting or validation examples'
    candidate_rows, selected, delta, bn_ids = [], None, None, None
    eligible = (len(groups['suspicious_fit']) >= args.min_suspicious and
                len(groups['suspicious_validation']) >= args.min_suspicious and
                len(groups['likely_clean_fit']) > 0 and len(groups['likely_clean_validation']) > 0)
    if eligible:
        branch = deepcopy(model)
        count = fit_balanced(branch, data, groups['likely_clean_fit'], groups['suspicious_fit'],
                             args, device, make_loader)
        delta = task_delta(base, branch)
        del branch
        torch.save(delta, run_dir / 'task_vector.pt')
        if any(not torch.isfinite(value).all() for value in delta.values()):
            raise ValueError('Nonfinite Option B task vector.')
        norm = sum(float(value.double().square().sum()) for value in delta.values()) ** .5
        log(run_dir, 'option_b_adaptation', examples_per_group=count, task_vector_l2=norm)
        bn_ids = np.random.default_rng(args.seed + 4300).permutation(groups['likely_clean_fit'])[:args.bn_samples]
        for alpha in args.alphas:
            apply_delta(model, base, delta, alpha)
            recalibrate_bn(model, make_loader(data, bn_ids, args.batch_size), device)
            row = {'alpha': alpha, **validation_metrics(model, data, groups, args, device, make_loader)}
            candidate_rows.append(row)
            log(run_dir, 'option_b_candidate', **row)
        zero = next(row for row in candidate_rows if row['alpha'] == 0)
        selected, reason = choose_candidate(raw_validation, zero, candidate_rows, args)
    # Freeze the decision before consulting any clean/triggered test results.
    selection = {'selected_alpha': selected['alpha'] if selected else 0., 'reason': reason,
                 'raw_base_validation': raw_validation, 'candidates': candidate_rows,
                 'selection_uses_test_data': False, 'selected_model': 'subtracted' if selected else 'raw_base'}
    (run_dir / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    model.load_state_dict(base)
    base_test = test_metrics(model, clean, bad, args, device)
    base_test['target_accuracy'] = test_target_accuracy(model, clean, args, device, make_loader)
    if selected:
        apply_delta(model, base, delta, selected['alpha'])
        recalibrate_bn(model, make_loader(data, bn_ids, args.batch_size), device)
    chosen_state = snapshot(model)
    torch.save(chosen_state, run_dir / 'final_model.pt')
    chosen_test = test_metrics(model, clean, bad, args, device)
    chosen_test['target_accuracy'] = test_target_accuracy(model, clean, args, device, make_loader)
    stress = {'description': 'One epoch of re-exposure to estimated suspicious and likely-clean fitting samples; does not use poison truth',
              'base': None, 'selected': None}
    if len(groups['suspicious_fit']) and len(groups['likely_clean_fit']):
        for name, weights in [('base', base), ('selected', chosen_state)]:
            model.load_state_dict(weights)
            fit_balanced(model, data, groups['likely_clean_fit'], groups['suspicious_fit'],
                         args, device, make_loader, stress=True)
            stress[name] = test_metrics(model, clean, bad, args, device)
            stress[name]['target_accuracy'] = test_target_accuracy(model, clean, args, device, make_loader)
        model.load_state_dict(chosen_state)
    else:
        stress['reason'] = 'Insufficient estimated fitting samples'
    (run_dir / 'reactivation.json').write_text(json.dumps(stress, indent=2) + '\n')
    return {**chosen_test, 'base_control': base_test, 'selected_alpha': selection['selected_alpha'],
            'selection_reason': reason, 'selection_used_test_data': False,
            'accuracy_delta_vs_base_pp': chosen_test['clean_accuracy'] - base_test['clean_accuracy'],
            'asr_delta_vs_base_pp': chosen_test['asr'] - base_test['asr'],
            'reactivation': stress}
