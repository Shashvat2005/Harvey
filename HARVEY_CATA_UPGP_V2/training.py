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
from artifacts import atomic_torch_save
from cata import CATAUnlearner
from refinement import snapshot, refine, write_json

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
                    torch.cuda.get_rng_state(device) if device.type == 'cuda' else torch.mps.get_rng_state() if device.type == 'mps' else None)
    atomic_torch_save(cpu_tree(state), path)


def restore_rng(state, device):
    a, b, c, d = state['rng']
    random.setstate(a)
    np.random.set_state(b)
    torch.set_rng_state(c)
    if d is not None and device.type == 'cuda':
        torch.cuda.set_rng_state(d, device)
    elif d is not None and device.type == 'mps':
        torch.mps.set_rng_state(d)


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


def holdout_ids(labels, seed, fraction=.05):
    rng=np.random.default_rng(seed+8100);out=[]
    for cls in range(10):
        ids=rng.permutation(np.flatnonzero(labels==cls))
        count=min(len(ids)-1,max(1,round(len(ids)*fraction))) if len(ids)>1 else 0
        out.extend(ids[:count])
    return np.asarray(sorted(out),dtype=np.int64)


def run(args):
    torch.set_num_threads(args.threads)
    device=torch.device(('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
                        if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Enable a GPU in Kaggle settings')
    if device.type=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS is unavailable')
    seed_all(args.seed)
    folder=ROOT/'runs'/args.run_name/args.variant/f'seed{args.seed}';folder.mkdir(parents=True,exist_ok=True)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py')
              if not any(part in ('runs','data','.venv','venv','__pycache__') for part in p.relative_to(ROOT).parts)}
    identity={k:v for k,v in vars(args).items() if k!='resume'}
    identity.update(resolved_device=str(device),torch=torch.__version__,
                    torchvision=sys.modules['torchvision'].__version__,numpy=np.__version__,
                    python=sys.version.split()[0],source_hashes=manifest)
    meta=folder/'metadata.json';checkpoint=folder/'checkpoint.pt'
    if meta.exists():
        if not args.resume:raise RuntimeError('Run exists; use --resume or a new run name')
        if json.loads(meta.read_text())!=identity:raise RuntimeError('Settings/code/device/environment changed; start a new named run')
        if (folder/'result.json').exists():print('Already complete:',folder);return
        if not checkpoint.exists():raise RuntimeError('No complete checkpoint yet; use a new run name')
    else:write_json(meta,identity)
    train,data,clean,bad,truth,labels=make_data(args)
    poison_hash=hashlib.sha256(truth.tobytes()).hexdigest();np.save(folder/'poison_ids.npy',truth)
    validation=holdout_ids(labels,args.seed);fit=np.setdiff1d(np.arange(len(data)),validation)
    model=new_model(args.seed,device)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False) if checkpoint.exists() and args.resume else {
        'phase':'warmup','epoch':0,'cycle':1,'elapsed':0.,'validation':validation,'fit':fit,'cata':None,'cata_cycles':0,'ga_cycles':0}
    if 'model' in state:model.load_state_dict(state['model']);restore_rng(state,device)
    start=time.monotonic();elapsed_before=state['elapsed']
    def event(name,**values):log(folder,name,**values)
    def commit():
        state['model']=snapshot(model);state['elapsed']=elapsed_before+time.monotonic()-start
        save(checkpoint,state,device)
        event('checkpoint_saved',phase=state['phase'],epoch=state.get('epoch'),cycle=state.get('cycle'))
    # Optional continuation reuses only a base whose original fitting/holdout
    # groups and poison identity are known. No original directory is modified.
    if state['phase']=='warmup' and state['epoch']==0 and args.variant=='harvey_option_b_v2' and args.reuse_source and args.profile=='full':
        source=Path(args.reuse_source).expanduser().resolve()
        required=['base_model.pt','option_b_groups.npz','metadata.json','poison_ids.npy']
        if not all((source/name).is_file() for name in required):raise FileNotFoundError('Reuse source missing required base/groups/metadata. Use --fresh for a new full run.')
        previous=json.loads((source/'metadata.json').read_text())
        if previous['seed']!=args.seed or previous['protocol']!=args.protocol or previous['profile']!='full':raise ValueError('Reuse source seed/protocol/profile mismatch')
        if not np.array_equal(np.load(source/'poison_ids.npy'),truth):raise ValueError('Poison IDs mismatch; refusing reuse')
        group_file=np.load(source/'option_b_groups.npz');groups={key:group_file[key] for key in group_file.files}
        for held in ('likely_clean_validation','suspicious_validation'):
            for fitting in ('likely_clean_fit','suspicious_fit','base_train'):
                if np.intersect1d(groups[held],groups[fitting]).size:raise ValueError('Reuse holdouts overlap fitting data')
        model.load_state_dict(torch.load(source/'base_model.pt',map_location='cpu',weights_only=True))
        state.update(phase='refine',groups=groups,reference_counts={'cata':0,'ga':0,'source':'reused_ordinary_HARVEY'},epoch=0)
        write_json(folder/'reuse_provenance.json',{'source':str(source),'base_sha256':hashlib.sha256((source/'base_model.pt').read_bytes()).hexdigest(),
                   'original_training_batch':previous['batch_size'],'seed':previous['seed'],'protocol':previous['protocol'],
                   'note':'Continuation of a previously trained Option B base, not a fresh independent replication.'})
        np.savez(folder/'groups.npz',**groups);atomic_torch_save(snapshot(model),folder/'base_model.pt');commit()
    if state['phase']=='warmup':
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        if 'optimizer' in state:opt.load_state_dict(state['optimizer'])
        for epoch in range(state['epoch'],args.warmup_epochs):
            loss=supervised(model,data,fit,args,device,opt)
            state.update(epoch=epoch+1,optimizer=opt.state_dict());event('warmup',epoch=epoch+1,loss=loss);commit()
        atomic_torch_save(cpu_tree(state),folder/'warmup_end_state.pt')
        warm=rce(model,data,args,device);order=np.argsort(warm);half=len(order)//2
        # Fixed held-out reference probes selected without true poison labels.
        clean_probe=validation[warm[validation]>=4.6052]
        poison_probe=validation[(labels[validation]==0)&(warm[validation]<=.5)]
        state.update(phase='reference',epoch=0,subset=order[:half],benign=order[half:],previous_clean=-1,
                     clean_probe=clean_probe,poison_probe=poison_probe)
        model=new_model(args.seed,device);state.pop('optimizer',None);commit()
    if state['phase']=='reference':
        engine=state.get('cata')
        while state['cycle']<=args.cycles:
            cycle=state['cycle']
            if state['epoch']==0 and 'next_reference' in state:model.load_state_dict(state['next_reference'])
            opt=torch.optim.Adam(model.parameters(),lr=.001)
            if 'optimizer' in state:opt.load_state_dict(state['optimizer'])
            ref_ids=np.setdiff1d(state['subset'],validation)
            if not len(ref_ids):raise RuntimeError('Reference subset is empty')
            for epoch in range(state['epoch'],args.reference_epochs):
                loss=supervised(model,data,ref_ids,args,device,opt)
                state.update(epoch=epoch+1,optimizer=opt.state_dict());event('reference',cycle=cycle,epoch=epoch+1,loss=loss);commit()
            poison_base=snapshot(model);forget=np.setdiff1d(state['benign'],validation)
            if len(forget):supervised(model,data,forget,args,device,opt,ascent=True)
            ga_state=snapshot(model);ga_losses=rce(model,data,args,device)
            selected_state=ga_state;selected_losses=ga_losses;next_reference=ga_state;choice='GA';score_best=-float('inf')
            cp,pp=state['clean_probe'],state['poison_probe']
            # No ground truth: candidate oracle separation and retention are
            # compared against the same cycle's ordinary GA update.
            if args.variant=='harvey_cata_upgp_v2' and len(forget):
                model.load_state_dict(poison_base)
                if engine is None:engine=CATAUnlearner(model,epochs=args.cata_epochs,scale=.1,weight_decay=0.)
                stats=engine.unlearn(model,loader(data,forget,args.batch_size,True));event('cata_request',cycle=cycle,**stats)
                for scale in args.cata_scales:
                    engine.scale=scale;engine.apply(model,base_state=poison_base)
                    candidate=rce(model,data,args,device)
                    valid=(len(cp)>=args.min_reference_probe and len(pp)>=args.min_reference_probe)
                    separation=float(candidate[cp].mean()-candidate[pp].mean()) if valid else None
                    ga_separation=float(ga_losses[cp].mean()-ga_losses[pp].mean()) if valid else None
                    retained=int((candidate>=4.6052).sum());ga_retained=int((ga_losses>=4.6052).sum())
                    protected=valid and float(candidate[pp].mean())<=float(ga_losses[pp].mean())+.1
                    accepted=(protected and retained>=ga_retained and separation>=ga_separation and separation>score_best)
                    event('reference_candidate',cycle=cycle,scale=scale,estimated_benign=retained,ga_estimated_benign=ga_retained,
                          separation=separation,ga_separation=ga_separation,accepted=accepted)
                    if accepted:
                        selected_state=snapshot(model);selected_losses=candidate;next_reference=poison_base
                        choice=f'CATA_{scale}';score_best=separation
            model.load_state_dict(selected_state)
            state['cata_cycles' if choice.startswith('CATA') else 'ga_cycles']+=1
            if cycle==1:state['first_reference']=snapshot(model)
            subset,benign=partition(selected_losses,labels);converged=len(benign)==state['previous_clean']
            event('split',cycle=cycle,reference_choice=choice,benign=len(benign),subset=len(subset))
            state.update(subset=subset,benign=benign,previous_clean=len(benign),losses=selected_losses,
                         next_reference=next_reference,cata=engine,cycle=cycle+1,epoch=0)
            state.pop('optimizer',None)
            if converged or cycle==args.cycles:state['phase']='meta'
            commit()
            if state['phase']=='meta':break
        del engine,opt
    if state['phase']=='meta':
        atomic_torch_save(cpu_tree(state),folder/'reference_end_state.pt')
        atomic_torch_save(snapshot(model),folder/'last_reference.pt');atomic_torch_save(state['first_reference'],folder/'first_reference.pt')
        pool=np.flatnonzero(state['losses']<4.6052);model.load_state_dict(state['first_reference']);first=rce(model,data,args,device)
        rest=pool[first[pool]<4.6052];recovered=pool[first[pool]>=4.6052];main=np.union1d(state['benign'],recovered)
        # High-confidence only in target-labelled rejected samples. Semantic
        # refinement below applies a separate teacher-consistency gate.
        suspicious=rest[(labels[rest]==0)&(first[rest]<=args.poison_rce)&(state['losses'][rest]<=args.poison_rce)]
        groups={'likely_clean_fit':np.setdiff1d(main,validation),'likely_clean_validation':np.intersect1d(main,validation),
                'suspicious_fit':np.setdiff1d(suspicious,validation),'suspicious_validation':np.intersect1d(suspicious,validation)}
        base_ids=(np.setdiff1d(np.arange(len(data)),np.union1d(suspicious,validation)) if args.variant=='harvey_option_b_v2'
                  else groups['likely_clean_fit'])
        groups['base_train']=base_ids
        if not len(base_ids):raise RuntimeError('Empty final fitting pool')
        # Truth is consumed only AFTER the sample-selection decision is fixed.
        hit=np.intersect1d(rest,truth).size
        write_json(folder/'split_quality.json',{'estimated_benign':len(main),'rejected':len(rest),
                   'precision':hit/len(rest) if len(rest) else None,'recall':hit/len(truth),
                   'base_train_poison_fraction':np.intersect1d(base_ids,truth).size/len(base_ids)})
        np.savez(folder/'groups.npz',**groups);np.savez(folder/'reference_scores.npz',first=first,last=state['losses'])
        counts={'cata':state['cata_cycles'],'ga':state['ga_cycles'],'source':'fresh_validation_guarded_reference'}
        state={'phase':'final','epoch':0,'elapsed':state['elapsed'],'groups':groups,'reference_counts':counts}
        model=new_model(args.seed,device);commit()
    if state['phase']=='final':
        opt=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=args.milestones)
        if 'optimizer' in state:opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        for epoch in range(state['epoch'],args.final_epochs):
            model.train();losses=[]
            for x,y in loader(train,state['groups']['base_train'],args.batch_size,True):
                opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x.to(device)),y.to(device))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite classifier loss')
                loss.backward();opt.step();losses.append(float(loss.detach()))
            sched.step();state.update(epoch=epoch+1,optimizer=opt.state_dict(),scheduler=sched.state_dict())
            event('final',epoch=epoch+1,loss=float(np.mean(losses)));commit()
        atomic_torch_save(cpu_tree(state),folder/'classifier_end_state.pt')
        atomic_torch_save(snapshot(model),folder/'base_model.pt');state.update(phase='refine',epoch=0)
        state.pop('optimizer',None);state.pop('scheduler',None);commit()
    if state['phase']=='refine':
        base,polish,chosen,selection=refine(model,data,labels,state['groups'],args,device,folder,loader,event)
        # Selection is persisted first. Test metrics cannot choose any candidate.
        stage_metrics={}
        for name,weights in [('base_control',base),('before_refinement',polish),('selected',chosen)]:
            model.load_state_dict(weights);stage_metrics[name]=metrics(model,clean,bad,args,device)
            target_ids=np.flatnonzero(clean.labels==0)
            with torch.no_grad():
                correct=0
                for x,y in loader(clean,target_ids,args.batch_size):correct+=int((model(x.to(device)).argmax(1).cpu()==y).sum())
            stage_metrics[name]['target_accuracy']=100*correct/len(target_ids)
        result={**stage_metrics['selected'],'base_control':stage_metrics['base_control'],
                'before_refinement':stage_metrics['before_refinement'],'selected_edit':selection['selected'],
                'eligible_for_refinement':selection['eligible'],'recovered_fitting_count':selection['recovered_fitting_count'],
                'recovery_enabled':selection['recovery_enabled'],'reference_counts':state['reference_counts'],
                'variant':args.variant,'seed':args.seed,'profile':args.profile,'protocol':args.protocol,
                'poison_hash':poison_hash,'synthetic':args.profile=='smoke','test_used_for_selection':False,
                'elapsed_seconds':elapsed_before+time.monotonic()-start,
                'runtime_excludes_reused_base_training':bool(args.reuse_source and args.profile=='full')}
        write_json(folder/'result.json',result);event('complete',**result)
        print('Results:',folder)


def parse_args():
    p=argparse.ArgumentParser(description='Conservative validation-selected HARVEY extensions V2')
    p.add_argument('--variant',choices=['harvey_option_b_v2','harvey_cata_upgp_v2','harvey_v2_control'],default='harvey_option_b_v2')
    p.add_argument('--profile',choices=['full','smoke'],default='full')
    p.add_argument('--protocol',choices=['release','paper-text'],default='release')
    p.add_argument('--seed',type=int,default=1);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--device',choices=['auto','cpu','mps','cuda:0'],default='auto')
    p.add_argument('--run-name',default='v2-full');p.add_argument('--resume',action='store_true')
    p.add_argument('--reuse-source');p.add_argument('--fresh',action='store_true')
    p.add_argument('--cata-scales',type=float,nargs='+',default=[.1,.3]);p.add_argument('--cata-epochs',type=int,default=2)
    p.add_argument('--min-reference-probe',type=int,default=16)
    p.add_argument('--poison-rce',type=float,default=.5);p.add_argument('--semantic-confidence',type=float,default=.85)
    p.add_argument('--min-suspicious',type=int,default=32);p.add_argument('--min-target-validation',type=int,default=20)
    p.add_argument('--polish-lrs',type=float,nargs='+',default=[.001,.005]);p.add_argument('--polish-epochs',type=int,default=2)
    p.add_argument('--direction-samples',type=int,default=1024);p.add_argument('--direction-lr',type=float,default=.001)
    p.add_argument('--direction-epochs',type=int,default=2);p.add_argument('--alphas',type=float,nargs='+',default=[0.,.25,.5,1.,2.])
    p.add_argument('--basis-samples',type=int,default=512);p.add_argument('--basis-batches',type=int,default=4)
    p.add_argument('--basis-energy',type=float,default=.99);p.add_argument('--upgp-lrs',type=float,nargs='+',default=[.00001,.0001])
    p.add_argument('--upgp-epochs',type=int,default=2);p.add_argument('--upgp-samples',type=int,default=1024)
    p.add_argument('--forget-weight',type=float,default=.1);p.add_argument('--max-relative-update',type=float,default=.002)
    p.add_argument('--target-tolerance',type=float,default=1.);p.add_argument('--probability-tolerance',type=float,default=.001)
    p.add_argument('--min-probability-gain',type=float,default=.0001)
    p.add_argument('--no-recover-clean',dest='recover_clean',action='store_false',default=True)
    p.add_argument('--recover-confidence',type=float,default=.95)
    a=p.parse_args()
    if Path(a.run_name).name!=a.run_name or a.run_name in ('.','..'):p.error('Run name must be a single folder name')
    if a.batch_size<2 or a.batch_size%2 or a.seed<0 or a.threads<1:p.error('Even batch size >=2, nonnegative seed and positive threads required')
    counts=[a.cata_epochs,a.min_reference_probe,a.min_suspicious,a.min_target_validation,a.direction_samples,a.direction_epochs,
            a.basis_samples,a.basis_batches,a.upgp_epochs,a.upgp_samples]
    if min(counts)<1 or a.polish_epochs<0:p.error('Invalid epoch/sample counts')
    floats=[*a.cata_scales,*a.polish_lrs,a.poison_rce,a.semantic_confidence,a.direction_lr,*a.alphas,a.basis_energy,
            *a.upgp_lrs,a.forget_weight,a.max_relative_update,a.target_tolerance,a.probability_tolerance,a.min_probability_gain,a.recover_confidence]
    if not np.isfinite(floats).all() or min(floats)<0:p.error('Numeric settings must be finite and nonnegative')
    if not 0<a.semantic_confidence<=1 or not 0<a.recover_confidence<=1 or not 0<a.basis_energy<=1 or not 0<=a.poison_rce<4.6052:p.error('Invalid confidence/energy/RCE')
    if min([*a.polish_lrs,*a.upgp_lrs,a.direction_lr,a.max_relative_update])<=0 or 0. not in a.alphas:p.error('Positive rates/update bound and alpha-zero control required')
    a.warmup_epochs,a.reference_epochs,a.cycles,a.final_epochs=20,14 if a.protocol=='release' else 10,20,100
    a.milestones=[60,80,90]
    if a.variant=='harvey_option_b_v2' and not a.fresh and not a.reuse_source and a.profile=='full':
        source=ROOT.parent/'HARVEY_OPTION_B/runs/option-b-full/harvey_option_b'/f'seed{a.seed}'
        if source.is_dir():a.reuse_source=str(source.resolve())
        else:p.error('No previous base for this seed. Set --fresh for full training or --reuse-source to its seed folder.')
    if a.fresh:a.reuse_source=None
    if a.variant!='harvey_option_b_v2' and a.reuse_source:p.error('Only Option B supports reuse of its saved base')
    if a.profile=='smoke':
        a.warmup_epochs=a.reference_epochs=a.cycles=a.final_epochs=1;a.milestones=[]
        a.batch_size=min(a.batch_size,8);a.cata_epochs=a.polish_epochs=a.direction_epochs=a.upgp_epochs=1
        a.direction_samples=a.basis_samples=a.upgp_samples=16;a.basis_batches=1
        a.min_reference_probe=a.min_suspicious=a.min_target_validation=1
    return a


if __name__=='__main__':run(parse_args())
