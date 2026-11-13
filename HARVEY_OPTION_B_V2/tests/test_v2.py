from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader,Subset
from refinement import snapshot,balanced_ids,semantic_filter,acceptable,better,clip_update,option_direction,refine,protect_target_direction,label_agreement_filter
from training import holdout_ids


def options(**kwargs):
    values=dict(seed=1,variant='harvey_option_b_v2',batch_size=4,semantic_confidence=.85,
                min_suspicious=1,min_target_validation=1,target_tolerance=1.,probability_tolerance=.001,
                min_probability_gain=.0001,max_relative_update=.002,polish_lrs=[.001],polish_epochs=1,
                direction_samples=4,direction_lr=.001,direction_epochs=1,alphas=[0.,.25,1.],
                basis_samples=4,basis_batches=1,basis_energy=.99,upgp_lrs=[.00001],upgp_epochs=1,
                upgp_samples=4,forget_weight=.1,recover_clean=True,recover_confidence=.95)
    return SimpleNamespace(**(values|kwargs))


def make_loader(dataset,ids,batch,shuffle=False):
    return DataLoader(Subset(dataset,list(map(int,ids))),batch_size=batch,shuffle=shuffle)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(1,2,1);self.bn=nn.BatchNorm2d(2);self.linear=nn.Linear(2,3)
        with torch.no_grad():
            self.linear.weight.zero_();self.linear.bias.copy_(torch.tensor([-3.,5.,-3.]))
    def forward(self,x):return self.linear(self.bn(self.conv(x)).mean((2,3)))


class V2(unittest.TestCase):
    def test_disjoint_stratified_holdout(self):
        y=np.arange(100)%10;ids=holdout_ids(y,1)
        self.assertEqual(len(np.unique(ids)),len(ids));self.assertEqual(len(ids),10)
        np.testing.assert_array_equal(ids,holdout_ids(y,1))

    def test_class_balancing(self):
        labels=np.array([0]*90+[1]*10)
        ids=balanced_ids(np.arange(100),labels,10000,1)
        self.assertTrue(.47<(labels[ids]==1).mean()<.53)

    def test_semantic_non_target_gate(self):
        data=TensorDataset(torch.randn(8,1,8,8),torch.zeros(8,dtype=torch.long));model=Tiny()
        ids,pseudo,_=semantic_filter(model,data,np.arange(8),make_loader,'cpu',.85)
        self.assertEqual(len(ids),8);self.assertTrue((pseudo==1).all())
        with torch.no_grad():model.linear.bias.copy_(torch.tensor([5.,-3.,-3.]))
        ids,_,_=semantic_filter(model,data,np.arange(8),make_loader,'cpu',.85)
        self.assertEqual(len(ids),0)

    def test_guards_prevent_accuracy_and_response_regression(self):
        anchor=dict(clean_accuracy=93.,target_accuracy=90.,target_count=30,suspicious_count=40,
                    suspicious_response=2.,suspicious_probability=.1)
        self.assertFalse(acceptable(anchor|{'clean_accuracy':92.99},anchor,options()))
        self.assertFalse(acceptable(anchor|{'suspicious_response':2.1},anchor,options()))
        self.assertFalse(better(anchor,anchor,anchor,options()))
        self.assertTrue(better(anchor|{'clean_accuracy':93.1},anchor,anchor,options()))

    def test_relative_update_bound(self):
        model=nn.Linear(3,2);base=snapshot(model)
        with torch.no_grad():
            for p in model.parameters():p.add_(10.)
        clip_update(model,base,.001)
        norm=sum(float(base[n].square().sum()) for n,_ in model.named_parameters())
        diff=sum(float((p.detach()-base[n]).square().sum()) for n,p in model.named_parameters())
        self.assertLessEqual((diff/norm)**.5,.001001)

    def test_frozen_bn_and_both_refinement_paths(self):
        torch.set_num_threads(2)
        labels=np.arange(24)%3
        labels[4:8]=0;labels[12:16]=0
        data=TensorDataset(torch.randn(24,1,8,8),torch.tensor(labels))
        groups=dict(likely_clean_fit=np.arange(4),suspicious_fit=np.arange(4,8),
                    likely_clean_validation=np.arange(8,12),suspicious_validation=np.arange(12,16))
        for variant in ['harvey_option_b_v2','harvey_cata_upgp_v2']:
            model=Tiny();base=snapshot(model)
            with TemporaryDirectory() as folder:
                _,_,_,selection=refine(model,data,labels,groups,options(variant=variant),'cpu',folder,make_loader,lambda *a,**k:None)
                self.assertTrue(selection['eligible']);self.assertFalse(selection['test_data_used'])
                self.assertTrue(selection['candidates'])
                self.assertTrue((Path(folder)/'selection.json').is_file())
                for n,b in model.named_buffers():torch.testing.assert_close(b,base[n],rtol=0,atol=0)

    def test_target_protection_removes_destructive_overlap(self):
        v={'p':torch.tensor([-2.,1.])};g={'p':torch.tensor([1.,0.])}
        protected=protect_target_direction(v,g)
        torch.testing.assert_close(protected['p'],torch.tensor([0.,1.]))
        self.assertGreaterEqual(float((protected['p']*g['p']).sum()),-1e-7)
        safe={'p':torch.tensor([2.,1.])}
        torch.testing.assert_close(protect_target_direction(safe,g)['p'],safe['p'])

    def test_label_agreement_recovery_only_matching_labels(self):
        model=Tiny();labels=np.array([0,1,2,1])
        data=TensorDataset(torch.randn(4,1,8,8),torch.tensor(labels))
        ids=label_agreement_filter(model,data,np.arange(4),labels,make_loader,'cpu',.95)
        np.testing.assert_array_equal(ids,np.array([1,3]))

    def test_overlap_rejected(self):
        data=TensorDataset(torch.randn(8,1,8,8),torch.zeros(8,dtype=torch.long))
        groups=dict(likely_clean_fit=np.array([1]),suspicious_fit=np.array([2]),
                    likely_clean_validation=np.array([1]),suspicious_validation=np.array([3]))
        with TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):refine(Tiny(),data,np.zeros(8,dtype=int),groups,options(),'cpu',folder,make_loader,lambda *a,**k:None)


if __name__=='__main__':unittest.main()
