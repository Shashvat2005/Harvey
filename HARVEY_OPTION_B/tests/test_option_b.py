from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from option_b import (make_groups, snapshot, task_delta, apply_delta, choose_candidate,
                      recalibrate_bn, fit_balanced, run_option_b)


def options(**changes):
    result = dict(seed=1, poison_rce=.5, clean_holdout=.1, poison_holdout=.2,
                  max_acc_drop=1., max_target_drop=3., min_response_drop=1.,
                  min_target_validation=2, min_suspicious=1, adapt_samples=8,
                  stress_samples=4, adapt_lr=.01, stress_lr=.001, adapt_epochs=1,
                  batch_size=4, bn_samples=16, alphas=[0., .5, 1.])
    return SimpleNamespace(**(result | changes))


class OptionBTests(unittest.TestCase):
    def test_holdouts_disjoint_and_uncertain_samples_retained(self):
        labels = np.arange(200) % 10
        labels[160:] = 0
        first, last = np.ones(200) * 8, np.ones(200) * 8
        first[160:180] = last[160:180] = .1
        first[180:] = last[180:] = 2.
        g = make_groups(np.arange(160), np.arange(160, 200), first, last, labels, options())
        np.testing.assert_array_equal(np.sort(g['high_confidence_poison']), np.arange(160, 180))
        self.assertTrue(set(range(180, 200)) <= set(g['base_train']))
        for held in ['likely_clean_validation', 'suspicious_validation']:
            for fit in ['base_train', 'likely_clean_fit', 'suspicious_fit']:
                self.assertEqual(len(np.intersect1d(g[held], g[fit])), 0)

    def test_fixed_anchor_and_buffers_not_subtracted(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        base = snapshot(model)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(.2)
            model[1].running_mean.fill_(100.)
        delta = task_delta(base, model)
        apply_delta(model, base, delta, .5)
        half = snapshot(model)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, base[name] - .1)
        torch.testing.assert_close(model[1].running_mean, base['1.running_mean'])
        apply_delta(model, base, delta, 1.)
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, base[name] - .2)
        apply_delta(model, base, delta, .5)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, half[name])

    def test_selection_uses_guards_and_raw_fallback(self):
        raw = dict(clean_accuracy=93., target_accuracy=94., suspicious_response=90.,
                   suspicious_mean_target_probability=.9, target_validation_count=20)
        rows = [dict(raw, alpha=0.), dict(raw, alpha=.1, suspicious_response=70.),
                dict(raw, alpha=.5, clean_accuracy=91.9, suspicious_response=10.),
                dict(raw, alpha=1., target_accuracy=90., suspicious_response=1.)]
        chosen, _ = choose_candidate(raw, raw, rows, options())
        self.assertEqual(chosen['alpha'], .1)
        chosen, _ = choose_candidate(raw, raw, [dict(raw, alpha=.5)], options())
        self.assertIsNone(chosen)
        chosen, _ = choose_candidate(dict(raw, target_validation_count=1), raw, rows, options())
        self.assertIsNone(chosen)

    def test_batchnorm_recalibration_changes_only_buffers(self):
        torch.manual_seed(1)
        model = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.Linear(3, 2))
        base = snapshot(model)
        data = DataLoader(TensorDataset(torch.randn(16, 3) + 3, torch.zeros(16, dtype=torch.long)), batch_size=4)
        recalibrate_bn(model, data, 'cpu')
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, base[name])
        self.assertFalse(torch.equal(model[1].running_mean, base['1.running_mean']))
        self.assertFalse(model.training)

    def test_equal_mixture_and_frozen_bn_during_adaptation(self):
        torch.manual_seed(4)
        model = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.Linear(3, 2))
        data = TensorDataset(torch.randn(16, 3), torch.arange(16) % 2)
        means = model[1].running_mean.clone()
        seen = []
        def make_loader(data, ids, batch):
            seen.extend(map(int, ids))
            return DataLoader(torch.utils.data.Subset(data, list(map(int, ids))), batch_size=batch)
        count = fit_balanced(model, data, np.arange(8), np.arange(8, 16), options(), 'cpu', make_loader)
        self.assertEqual(count, 8)
        self.assertEqual(sum(i < 8 for i in seen), sum(i >= 8 for i in seen))
        torch.testing.assert_close(model[1].running_mean, means)

    def test_postprocessing_integration_selects_before_test_access(self):
        torch.manual_seed(2)
        model = nn.Linear(3, 2)
        data = TensorDataset(torch.randn(24, 3), torch.arange(24) % 2)
        class Clean(TensorDataset):
            pass
        clean = Clean(torch.randn(8, 3), torch.arange(8) % 2)
        clean.labels = np.arange(8) % 2
        groups = dict(likely_clean_fit=np.arange(8), suspicious_fit=np.arange(8, 12),
                      likely_clean_validation=np.arange(12, 20), suspicious_validation=np.arange(20, 24))
        def make_loader(dataset, ids, batch):
            return DataLoader(torch.utils.data.Subset(dataset, list(map(int, ids))), batch_size=batch)
        calls = []
        with TemporaryDirectory() as name:
            folder = Path(name)
            def evaluate(*unused):
                self.assertTrue((folder / 'selection.json').exists())
                calls.append(True)
                return dict(clean_accuracy=50., asr=50.)
            result = run_option_b(model, data, clean, clean, groups, options(), 'cpu', folder,
                                  make_loader, evaluate, lambda *a, **k: None)
            self.assertTrue((folder / 'task_vector.pt').exists())
            self.assertTrue((folder / 'final_model.pt').exists())
            self.assertTrue((folder / 'reactivation.json').exists())
            self.assertFalse(result['selection_used_test_data'])
            self.assertEqual(len(calls), 4)
            self.assertIn(result['selected_alpha'], [0., .5, 1.])


if __name__ == '__main__':
    unittest.main()
