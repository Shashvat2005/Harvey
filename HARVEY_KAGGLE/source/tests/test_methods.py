import unittest
import torch
from cata import ConflictAverseMemory, topk_mask, CATAUnlearner, snapshot
from upgp import basis_from_rows, project, build_bases, unlearning_loss
from torch.utils.data import DataLoader, TensorDataset


class Methods(unittest.TestCase):
    def test_sign_election_and_zero_ties(self):
        memory = ConflictAverseMemory(3)
        memory.add(torch.tensor([2., 1., -2.]))
        memory.add(torch.tensor([-1., -1., -4.]))
        torch.testing.assert_close(memory.aggregate(), torch.tensor([2., 0., -3.]))

    def test_global_topk_keeps_ties(self):
        torch.testing.assert_close(topk_mask(torch.tensor([1., -2., 2., .1]), .25),
                                   torch.tensor([0., -2., 2., 0.]))

    def test_projection_annihilates_full_row_space(self):
        rows = [torch.tensor([[1., 0., 0.], [0., 2., 0.]])]
        vectors, weights = basis_from_rows(rows, energy=1.)
        # Hard projection is orthogonal; relaxed Eq.18 does not promise exact zero.
        g = torch.tensor([[3., 4., 5.]])
        result = project(g, (vectors, torch.ones_like(weights)))
        torch.testing.assert_close(result, torch.tensor([[0., 0., 5.]]))
        torch.testing.assert_close(project(torch.ones(3), None), torch.ones(3))

    def test_relaxed_projection_matches_explicit_matrix(self):
        torch.manual_seed(4)
        rows = [torch.randn(4, 3), torch.randn(4, 3)]
        v, weights = basis_from_rows(rows)
        g = torch.randn(2, 3)
        expected = g @ (torch.eye(3) - v @ torch.diag(weights) @ v.T)
        torch.testing.assert_close(project(g, (v, weights)), expected)

    def test_cata_preserves_base_buffers_and_learning(self):
        torch.manual_seed(3)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 2))
        dataset = TensorDataset(torch.randn(8, 4), torch.arange(8) % 2)
        engine = CATAUnlearner(model, epochs=1, lr=.01)
        base = snapshot(model)
        engine.unlearn(model, DataLoader(dataset, batch_size=4))
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, base[name])
        engine.restore_learning_state(model)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, base[name])

    def test_upgp_forward_backward(self):
        torch.manual_seed(2)
        model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, 2))
        x, y = torch.randn(8, 4), torch.arange(8) % 2
        bases = build_bases(model, DataLoader(TensorDataset(x, y), batch_size=4), 'cpu', batches=2)
        loss = unlearning_loss(model, (x, y), (x, y), 'cpu')
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.isfinite(project(parameter.grad, bases.get(name))).all())


if __name__ == '__main__':
    unittest.main()
