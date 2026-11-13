"""Validation-selected edits; no poison truth or test images are accepted here."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from artifacts import atomic_torch_save
from upgp import build_bases, project


def snapshot(model):
    return {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}


def write_json(path, value):
    path = Path(path); temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def balanced_ids(ids, labels, count, seed):
    ids = np.asarray(ids, dtype=np.int64)
    if not len(ids):
        return ids
    counts = np.bincount(labels[ids], minlength=10)
    weights = 1. / counts[labels[ids]].clip(1)
    weights /= weights.sum()
    return np.random.default_rng(seed).choice(ids, size=count, replace=True, p=weights)


@torch.no_grad()
def semantic_filter(model, data, ids, make_loader, device, confidence=.85):
    """Two weak views must agree on a confident NON-target semantic label.

    No known trigger mask or true poison membership is inspected. The target
    label is 0, matching the declared all-to-one benchmark setup.
    """
    model.eval(); chosen=[]; pseudo=[]; probabilities=[]
    ids = np.asarray(ids, dtype=np.int64)
    offset=0
    for x, _ in make_loader(data, ids, 128):
        x=x.to(device); h,w=x.shape[-2:]
        a=model(x.flip(-1)).softmax(1)
        padded=F.pad(x,(2,2,2,2),mode='reflect')
        b=model(padded[:,:,0:h,4:4+w]).softmax(1)
        ca,ya=a.max(1); cb,yb=b.max(1)
        keep=((ya==yb)&(ya!=0)&(ca>=confidence)&(cb>=confidence)).cpu().numpy()
        chosen.extend(ids[offset:offset+len(x)][keep].tolist())
        pseudo.extend(ya.cpu().numpy()[keep].tolist())
        probabilities.extend(torch.minimum(ca,cb).cpu().numpy()[keep].tolist())
        offset+=len(x)
    return np.asarray(chosen,dtype=np.int64),np.asarray(pseudo,dtype=np.int64),probabilities


@torch.no_grad()
def label_agreement_filter(model,data,ids,labels,make_loader,device,confidence=.95):
    """Recover FITTING images only when original and both weak views agree
    confidently with their OBSERVED label. No correction of unknown labels."""
    model.eval();ids=np.asarray(ids,dtype=np.int64);out=[];offset=0
    for x,_ in make_loader(data,ids,128):
        x=x.to(device);h,w=x.shape[-2:];padded=F.pad(x,(2,2,2,2),mode='reflect')
        expected=torch.tensor(labels[ids[offset:offset+len(x)]],device=device)
        keep=torch.ones(len(x),device=device,dtype=torch.bool)
        for view in (x,x.flip(-1),padded[:,:,0:h,4:4+w]):
            confidence_value,pred=model(view).softmax(1).max(1)
            keep&=(pred==expected)&(confidence_value>=confidence)
        out.extend(ids[offset:offset+len(x)][keep.cpu().numpy()].tolist());offset+=len(x)
    return np.asarray(out,dtype=np.int64)


def target_retention_gradient(model,data,ids,labels,args,make_loader,device):
    target=np.asarray(ids)[labels[np.asarray(ids)]==0]
    target=np.random.default_rng(args.seed+7700).permutation(target)[:128]
    model.eval();model.zero_grad(set_to_none=True)
    if not len(target):return None
    for x,y in make_loader(data,target,args.batch_size):
        loss=F.cross_entropy(model(x.to(device)),y.to(device))*(len(y)/len(target))
        loss.backward()
    gradient={name:p.grad.detach().cpu().clone() for name,p in model.named_parameters()}
    model.zero_grad(set_to_none=True);return gradient


def protect_target_direction(vector,gradient):
    """First-order protection of ESTIMATED clean target-class fitting loss.
    Subtraction uses -vector, so a negative <vector, grad> is destructive."""
    if gradient is None:return {n:v.clone() for n,v in vector.items()}
    dot=sum(float((vector[n].double()*gradient[n].double()).sum()) for n in vector)
    norm=sum(float(g.double().square().sum()) for g in gradient.values())
    coefficient=min(dot,0.)/max(norm,1e-20)
    return {n:v-coefficient*gradient[n] for n,v in vector.items()}


@torch.no_grad()
def validate(model, data, clean_ids, suspicious_ids, make_loader, device, batch):
    model.eval(); correct=total=target_correct=target_total=pred_target=sus_total=0; prob=0.
    for x,y in make_loader(data,clean_ids,batch):
        prediction=model(x.to(device)).argmax(1).cpu()
        correct+=int((prediction==y).sum());total+=len(y)
        target_correct+=int(((prediction==y)&(y==0)).sum());target_total+=int((y==0).sum())
    for x,_ in make_loader(data,suspicious_ids,batch):
        p=model(x.to(device)).softmax(1)
        pred_target+=int((p.argmax(1)==0).sum());prob+=float(p[:,0].sum());sus_total+=len(x)
    return {'clean_accuracy':100*correct/total if total else None,
            'target_accuracy':100*target_correct/target_total if target_total else None,
            'target_count':target_total,'suspicious_count':sus_total,
            'suspicious_response':100*pred_target/sus_total if sus_total else None,
            'suspicious_probability':prob/sus_total if sus_total else None}


def acceptable(candidate, anchor, args):
    """No accepted drop in overall accuracy; target loss and response are capped."""
    keys=('clean_accuracy','target_accuracy','suspicious_response','suspicious_probability')
    if any(candidate.get(k) is None or anchor.get(k) is None for k in keys): return False
    if anchor['target_count']<args.min_target_validation or anchor['suspicious_count']<args.min_suspicious: return False
    return (candidate['clean_accuracy']>=anchor['clean_accuracy']-1e-8
            and candidate['target_accuracy']>=anchor['target_accuracy']-args.target_tolerance-1e-8
            and candidate['suspicious_response']<=anchor['suspicious_response']+1e-8
            and candidate['suspicious_probability']<=anchor['suspicious_probability']+args.probability_tolerance)


def better(candidate, incumbent, anchor, args):
    if not acceptable(candidate,anchor,args): return False
    # Accuracy gain is prioritized under the retention/response guards. Ties
    # need lower response or lower target confidence; no forced nonzero edit.
    if candidate['clean_accuracy']>incumbent['clean_accuracy']+1e-8: return True
    if abs(candidate['clean_accuracy']-incumbent['clean_accuracy'])<1e-8:
        if candidate['suspicious_response']<incumbent['suspicious_response']-1e-8: return True
        return (candidate['suspicious_response']<=incumbent['suspicious_response']+1e-8
                and candidate['suspicious_probability']<incumbent['suspicious_probability']-args.min_probability_gain)
    return False


def clean_loss(model, teacher, x, y, kd=.5):
    logits=model(x)
    with torch.no_grad(): soft=teacher(x).softmax(1)
    return F.cross_entropy(logits,y)+kd*F.kl_div(logits.log_softmax(1),soft,reduction='batchmean')


def clip_update(model, anchor, max_relative):
    parameters=dict(model.named_parameters())
    total=sum(float(anchor[n].square().sum()) for n in parameters)
    delta=sum(float((p.detach()-anchor[n].to(p.device)).square().sum()) for n,p in parameters.items())
    ratio=(delta/max(total,1e-20))**.5
    if ratio>max_relative:
        with torch.no_grad():
            for name,p in parameters.items():
                base=anchor[name].to(p.device)
                p.copy_(base+(p-base)*(max_relative/ratio))
    return ratio


def fine_tune(model, teacher, data, ids, labels, args, make_loader, device, lr, epochs, callback, seed):
    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9,weight_decay=0.)
    anchor=snapshot(model)
    for epoch in range(epochs):
        # Trainable BN affine parameters remain eligible; running buffers stay fixed.
        model.eval();order=balanced_ids(ids,labels,len(ids),seed+epoch)
        for x,y in make_loader(data,order,args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss=clean_loss(model,teacher,x.to(device),y.to(device))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite clean-polish loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        clip_update(model,anchor,args.max_relative_update)
        callback(epoch+1)


def option_direction(model, data, clean_ids, suspect_ids, labels, args, make_loader, device, mixed=True):
    """Matched anchored branches: mixture learning vs ordinary clean learning."""
    rng=np.random.default_rng(args.seed+7100)
    n=min(len(clean_ids),len(suspect_ids),args.direction_samples)
    if n<1: raise ValueError('No samples for Option B adaptation')
    clean=balanced_ids(clean_ids,labels,n,args.seed+7110)
    suspect=rng.permutation(suspect_ids)[:n]
    opt=torch.optim.SGD(model.parameters(),lr=args.direction_lr)
    model.eval()
    for _ in range(args.direction_epochs):
        for benign,suspicious in zip(make_loader(data,clean,args.batch_size//2),
                                      make_loader(data,suspect,args.batch_size//2)):
            x=torch.cat([benign[0],suspicious[0] if mixed else benign[0]]).to(device)
            y=torch.cat([benign[1],suspicious[1] if mixed else benign[1]]).to(device)
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x),y)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite Option B direction loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()


def refine(model, data, labels, groups, args, device, folder, make_loader, log):
    """Returns frozen selected state; caller evaluates test data afterward only."""
    for held in ('likely_clean_validation','suspicious_validation'):
        for fitting in ('likely_clean_fit','suspicious_fit'):
            if np.intersect1d(groups[held],groups[fitting]).size:
                raise ValueError('Validation and fitting groups overlap')
    folder=Path(folder);base=snapshot(model);teacher=deepcopy(model).eval()
    raw_fit=np.asarray(groups['suspicious_fit']);raw_val=np.asarray(groups['suspicious_validation'])
    fit,pseudo,_=semantic_filter(teacher,data,raw_fit,make_loader,device,args.semantic_confidence)
    val,_,_=semantic_filter(teacher,data,raw_val,make_loader,device,args.semantic_confidence)
    clean_fit=np.asarray(groups['likely_clean_fit']);clean_val=np.asarray(groups['likely_clean_validation'])
    recovered=(label_agreement_filter(teacher,data,raw_fit,labels,make_loader,device,args.recover_confidence)
               if args.recover_clean else np.array([],dtype=np.int64))
    clean_fit=np.union1d(clean_fit,recovered)
    if np.intersect1d(clean_fit,fit).size:raise RuntimeError('Semantic and label-agreement fitting groups overlap')
    np.savez(folder/'refinement_groups.npz',suspect_fit=fit,suspect_validation=val,pseudo_labels=pseudo,
             clean_fit=clean_fit,clean_validation=clean_val,recovered_fit=recovered)
    anchor=validate(model,data,clean_val,val,make_loader,device,args.batch_size)
    best=dict(anchor);best_state=base;selected={'kind':'base','stage':'base'};records=[]
    eligible=(len(fit)>=args.min_suspicious and len(val)>=args.min_suspicious
              and anchor['target_count']>=args.min_target_validation and args.variant!='harvey_v2_control')
    log('refinement_groups',raw_suspicious_fit=len(raw_fit),filtered_suspicious_fit=len(fit),
        raw_suspicious_validation=len(raw_val),filtered_suspicious_validation=len(val),recovered_fitting_images=len(recovered),eligible=eligible)

    def consider(kind,**config):
        nonlocal best,best_state,selected
        measurements=validate(model,data,clean_val,val,make_loader,device,args.batch_size)
        accepted=eligible and better(measurements,best,anchor,args)
        number=len(records)+1
        relative=f'candidates/{number:03d}-{kind}.pt'
        atomic_torch_save(snapshot(model),folder/relative)
        row={'kind':kind,**config,**measurements,'accepted':accepted,'model_file':relative}
        records.append(row);write_json(folder/'candidates_so_far.json',records);log('candidate',**row)
        if accepted: best=measurements;best_state=snapshot(model);selected={'kind':kind,**config}

    # Utility stage is its own ablation; each LR arm restarts from the raw base.
    if eligible and len(clean_fit) and args.polish_epochs:
        for lr in args.polish_lrs:
            model.load_state_dict(base)
            fine_tune(model,teacher,data,clean_fit,labels,args,make_loader,device,lr,args.polish_epochs,
                      lambda epoch:consider('clean_polish',lr=lr,epoch=epoch),args.seed+7200)
    polish=best_state;polish_choice=dict(selected);polish_validation=dict(best)
    atomic_torch_save(polish,folder/'pre_refinement.pt')
    if eligible and args.variant=='harvey_option_b_v2':
        # Both literal Option B and differential mixture-minus-clean directions
        # are ablated from the SAME chosen utility-stage anchor.
        model.load_state_dict(polish)
        branch=deepcopy(model);option_direction(branch,data,clean_fit,fit,labels,args,make_loader,device,True)
        mixed={n:p.detach().cpu()-polish[n] for n,p in branch.named_parameters()};del branch
        branch=deepcopy(model);option_direction(branch,data,clean_fit,fit,labels,args,make_loader,device,False)
        differential={n:mixed[n]-(p.detach().cpu()-polish[n]) for n,p in branch.named_parameters()};del branch
        model.load_state_dict(polish)
        retention=target_retention_gradient(model,data,groups['likely_clean_fit'],labels,args,make_loader,device)
        protected_mixed=protect_target_direction(mixed,retention)
        protected_differential=protect_target_direction(differential,retention)
        vectors={'mixed':mixed,'differential':differential,
                 'target_protected_mixed':protected_mixed,'target_protected_differential':protected_differential}
        atomic_torch_save(vectors,folder/'task_vectors.pt')
        for name,vector in vectors.items():
            for alpha in args.alphas:
                if alpha==0:continue
                model.load_state_dict(polish)
                with torch.no_grad():
                    for n,p in model.named_parameters():p.add_(vector[n].to(device),alpha=-alpha)
                unclipped=clip_update(model,polish,args.max_relative_update)
                consider('option_b',direction=name,alpha=alpha,unclipped_relative_update=unclipped)
    elif eligible and args.variant=='harvey_cata_upgp_v2':
        model.load_state_dict(polish)
        ids=balanced_ids(clean_fit,labels,min(args.basis_samples,len(clean_fit)),args.seed+7300)
        bases=build_bases(model,make_loader(data,ids,args.batch_size),device,
                          batches=args.basis_batches,energy=args.basis_energy,alpha=10.)
        bases={n:tuple(v.to(device) for v in basis) for n,basis in bases.items()}
        pseudo_map=dict(zip(fit.tolist(),pseudo.tolist()))
        for lr in args.upgp_lrs:
            model.load_state_dict(polish);opt=torch.optim.SGD(model.parameters(),lr=lr)
            for epoch in range(args.upgp_epochs):
                model.eval();rng=np.random.default_rng(args.seed+7400+epoch)
                sus=rng.permutation(fit)[:args.upgp_samples]
                normals=balanced_ids(clean_fit,labels,len(sus),args.seed+7500+epoch)
                for (x,y),(xp,_) in zip(make_loader(data,normals,args.batch_size),make_loader(data,sus,args.batch_size)):
                    x,y,xp=x.to(device),y.to(device),xp.to(device)
                    offset=getattr(opt,'_sample_offset',0)
                    target=torch.tensor([pseudo_map[int(i)] for i in sus[offset:offset+len(xp)]],device=device)
                    opt._sample_offset=offset+len(xp)
                    opt.zero_grad(set_to_none=True)
                    retention=clean_loss(model,teacher,x,y,kd=1.)
                    params=list(model.parameters())
                    clean_grad=torch.autograd.grad(retention,params)
                    logits=model(xp);masked=logits.clone();masked[:,0]=-torch.inf
                    negative=(torch.logsumexp(logits,1)-torch.logsumexp(masked,1)).mean()
                    forget=.1*negative+.5*F.cross_entropy(logits,target)
                    forget_grad=torch.autograd.grad(forget,params)
                    for (name,p),gc,gf in zip(model.named_parameters(),clean_grad,forget_grad):
                        projected=project(gf,bases.get(name))
                        # Remove remaining first-order conflict with this clean batch.
                        dot=(projected*gc).sum()
                        if dot<0:projected=projected-dot/(gc.square().sum()+1e-12)*gc
                        p.grad=gc+args.forget_weight*projected
                    torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
                opt._sample_offset=0
                unclipped=clip_update(model,polish,args.max_relative_update)
                consider('upgp',lr=lr,epoch=epoch+1,unclipped_relative_update=unclipped)
    model.load_state_dict(best_state)
    # Finite weights and exact fixed buffers are execution invariants.
    for name,value in model.named_buffers():
        if not torch.equal(value.detach().cpu(),base[name]):raise RuntimeError('BN buffers changed during refinement')
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('Nonfinite selected parameters')
    selection={'base_validation':anchor,'utility_stage_choice':polish_choice,
               'utility_stage_validation':polish_validation,'selected':selected,'selected_validation':best,
               'eligible':eligible,'recovered_fitting_count':len(recovered),'recovery_enabled':args.recover_clean,
               'candidates':records,'test_data_used':False,
               'note':'Validation proxies are not proof of true poison identity or lower held-out ASR.'}
    write_json(folder/'selection.json',selection);atomic_torch_save(best_state,folder/'final_model.pt')
    del teacher
    return base,polish,best_state,selection
