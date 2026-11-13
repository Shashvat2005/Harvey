import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.cata import CATAUnlearner, ConflictAverseMemory, snapshot, topk_mask


def loader():
    # Deterministic two-class data, no downloads or accelerator required.
    return DataLoader(TensorDataset(torch.tensor([[2., 0.], [0., 2.],
                                                  [1., 0.], [0., 1.]]),
                                    torch.tensor([0, 1, 0, 1])), batch_size=2)


class CATATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_global_threshold_and_ties(self):
        value = torch.tensor([1., -4., 3., -3., 0.])
        torch.testing.assert_close(topk_mask(value, .4), torch.tensor([0., -4., 3., -3., 0.]))
        torch.testing.assert_close(topk_mask(value, 1.), value)
        torch.testing.assert_close(topk_mask(torch.zeros(4), .3), torch.zeros(4))
        for ratio in [0., -1., 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                topk_mask(value, ratio)

    def test_sign_vote_uses_magnitude_not_majority_and_excludes_zeros(self):
        memory = ConflictAverseMemory(4)
        for value in [[3., 2., 0., 1.], [-1., -2., 0., 3.], [-1., 0., 4., -1.]]:
            memory.add(torch.tensor(value))
        torch.testing.assert_close(memory.aggregate(), torch.tensor([3., 0., 4., 2.]))
        self.assertEqual(memory.requests, 3)

    def test_streaming_memory_matches_explicit_paper_equations(self):
        vectors = torch.randn(13, 37)
        vectors[vectors.abs() < .7] = 0
        memory = ConflictAverseMemory(37)
        for step, vector in enumerate(vectors):
            memory.add(vector)
            history = vectors[:step + 1]
            vote = history.sum(0).sign()
            consistent = (history != 0) & (history.sign() == vote)
            expected = (history * consistent).sum(0) / consistent.sum(0).clamp_min(1)
            torch.testing.assert_close(memory.aggregate(), expected)

    def test_task_is_negative_of_adaptation_from_fixed_anchor(self):
        model = nn.Linear(2, 2)
        engine = CATAUnlearner(model, keep_ratio=1., epochs=2, lr=.02, weight_decay=0)
        anchor = snapshot(model)
        branch = deepcopy(model)
        optimizer = torch.optim.AdamW(branch.parameters(), lr=.02, weight_decay=0)
        for _ in range(2):
            for inputs, labels in loader():
                optimizer.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(branch(inputs), labels).backward()
                optimizer.step()
        stats = engine.unlearn(model, loader())
        for name, parameter in model.named_parameters():
            expected = anchor[name] - .7 * (dict(branch.named_parameters())[name] - anchor[name])
            torch.testing.assert_close(parameter, expected)
        self.assertEqual(stats['examples_seen'], 8)
        first = snapshot(model)
        engine.restore_learning_state(model)
        engine.unlearn(model, loader())
        # Identical requests are averaged, never summed or applied twice.
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, first[name])
        for name, value in engine.anchor.items():
            torch.testing.assert_close(value, anchor[name])

    def test_harvey_poison_learning_is_preserved_and_not_compounded(self):
        model = nn.Linear(2, 2)
        engine = CATAUnlearner(model, epochs=1, lr=.02, weight_decay=0)
        engine.unlearn(model, loader())
        engine.restore_learning_state(model)
        with torch.no_grad():
            model.weight.add_(.5)  # represents next round's poison learning
        poison_state = snapshot(model)
        engine.unlearn(model, loader())
        offset = 0
        correction = engine.memory.aggregate()
        for name, parameter in model.named_parameters():
            delta = correction[offset:offset + parameter.numel()].view_as(parameter)
            torch.testing.assert_close(parameter, poison_state[name] + .7 * delta)
            offset += parameter.numel()
        engine.restore_learning_state(model)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, poison_state[name])
        # Pure paper update is also available, explicitly anchored at theta0.
        engine.apply(model)
        offset = 0
        for name, parameter in model.named_parameters():
            delta = correction[offset:offset + parameter.numel()].view_as(parameter)
            torch.testing.assert_close(parameter, engine.anchor[name] + .7 * delta)
            offset += parameter.numel()

    def test_buffers_and_frozen_parameters_are_not_forget_branch_updates(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2), nn.Linear(2, 2))
        model[0].weight.requires_grad_(False)
        engine = CATAUnlearner(model, epochs=1, lr=.02)
        before = snapshot(model)
        engine.unlearn(model, loader())
        after = snapshot(model)
        for name in ['0.weight', '1.running_mean', '1.running_var', '1.num_batches_tracked']:
            torch.testing.assert_close(after[name], before[name])
        self.assertEqual(after['1.num_batches_tracked'].dtype, torch.int64)

    def test_empty_request_and_failed_request_do_not_add_history(self):
        model = nn.Linear(2, 2)
        engine = CATAUnlearner(model, epochs=1)
        before = snapshot(model)
        empty = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
        engine.unlearn(model, empty)
        self.assertEqual(engine.memory.requests, 0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])
        bad = DataLoader(TensorDataset(torch.full((2, 2), float('nan')), torch.tensor([0, 1])))
        with self.assertRaises(ValueError):
            engine.unlearn(model, bad)
        self.assertEqual(engine.memory.requests, 0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_both_supplied_configs_select_cata(self):
        from config import cfg_from_file
        for attack in ['badnets', 'blend']:
            cfg = cfg_from_file('conf/ResNet18_cifar10_SL_' + attack + '_rate010.yml')
            self.assertEqual(cfg.UNLEARNING, 'CATA')
            self.assertEqual(cfg.CATA.KEEP_RATIO, .3)
            self.assertEqual(cfg.CATA.EPOCHS, 4)

    def test_stage2_routes_to_cata_and_excludes_warmup_and_final_model(self):
        from config import cfg_from_file
        import train_test
        cfg = deepcopy(cfg_from_file('conf/ResNet18_cifar10_SL_badnets_rate010.yml'))
        cfg.BACKDOOR.ref_c1epochs = cfg.BACKDOOR.ref_epochs = 0
        cfg.BACKDOOR.warmup_epochs = cfg.OPTIMIZER_BACKBONE.EPOCH_TL = 0
        data = loader()
        data.dataset.transform = None
        loaders = {name: data for name in ['train', 'train_clean', 'train_unlabel', 'test', 'test_bad']}
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        state = {}
        with patch.object(train_test, 'test', return_value=50.), \
             patch.object(train_test, 'train_epoch', side_effect=AssertionError('legacy GA called')):
            for cycle in [1, 2]:
                train_test.train_model({'backbone': model}, 'RCE', nn.CrossEntropyLoss(),
                                       {'backbone': optimizer}, {}, loaders, 'cpu', cfg,
                                       cycle, False, 2, ref_train=True, cata_state=state)
            self.assertEqual(state['engine'].memory.requests, 2)
            history = state['engine'].memory.requests
            train_test.train_model({'backbone': model}, 'RCE', nn.CrossEntropyLoss(),
                                   {'backbone': optimizer}, {}, loaders, 'cpu', cfg,
                                   0, False, 2, warmup_train=True, ref_train=True)
            train_test.train_model({'backbone': model}, 'RCE', nn.CrossEntropyLoss(),
                                   {'backbone': optimizer}, {}, loaders, 'cpu', cfg,
                                   0, False, 2, ref_train=False)
            self.assertEqual(state['engine'].memory.requests, history)

    def test_invalid_hyperparameters(self):
        for options in [{'epochs': 0}, {'lr': 0}, {'keep_ratio': 2},
                        {'scale': float('nan')}, {'weight_decay': -1}]:
            with self.assertRaises(ValueError):
                CATAUnlearner(nn.Linear(2, 2), **options)


if __name__ == '__main__':
    unittest.main()
