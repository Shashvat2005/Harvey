from config import *
from utils.loss import *
from utils.mixmatch import *
from utils.AverageMeter import AverageMeter
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import time

from torch.nn.utils import clip_grad_norm_


# Loss Prediction Loss from TA_VAAL, ranking loss not the absolute loss in lloss 
def LossPredLoss_ta(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = torch.nn.BCELoss()
    input = (input - input.flip(0))[:len(input)//2]
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()
    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0))
    
    if reduction == 'mean':
        loss = criterion(diff,one)
    elif reduction == 'none':
        loss = criterion(diff,one)
    else:
        NotImplementedError()
    
    return loss


def test(models, method, dataloaders, device, mode='val'):
    assert mode in ['val', 'test', 'test_bad', 'train', 'train_unlabel']
    models['backbone'].eval()
    if method == 'lloss':
        models['module'].eval()
    
    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            with torch.cuda.device(device):
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            scores = models['backbone'](inputs)
            #scores = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    #print('total number for test {}'.format(total))
    return 100 * correct / total


def calc_loss(models, dataloaders, criterion, device, mode):
    models['backbone'].eval()
    losses = []
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            with torch.cuda.device(device):
                inputs = inputs.cuda()
                labels = labels.cuda()
        
            scores = models['backbone'](inputs)
            target_loss = criterion(scores, labels)
            losses.append(target_loss)
    l = torch.cat(losses)
    return torch.sum(l)/len(l)


iters = 0


def train_epoch(models, method, criterion, optimizers, schedulers, dataloaders, epoch, epoch_loss, device, MARGIN,
                WEIGHT, log_data_map, cfgs, warmup_train=False, VIS=False, Add_Noise=False, LGA=False,
                Unlearn_Clean=False, Unlearn_Gamma=0.0001, cycle=0):
    models['backbone'].train()

    global iters

    if Unlearn_Clean:
        train_loader = dataloaders['train_clean']
    else:
        train_loader = dataloaders['train']

    total_t = 0
    correct_t = 0
    for (batch_idx, data) in enumerate(train_loader):
        with torch.cuda.device(device):
            inputs = data[0].cuda()

            if Add_Noise:
                inputs = inputs + torch.empty_like(inputs).uniform_(-0.01, 0.01)
                inputs = torch.clamp(inputs, min=inputs.min(), max=inputs.max())

            labels = data[1].cuda()

        iters += 1

        optimizers['backbone'].zero_grad()
        if method == 'lloss' or method == 'TA-VAAL':
            optimizers['module'].zero_grad()
                   
        scores = models['backbone'](inputs)

        #scores = models['backbone'](inputs)
        if log_data_map:
            log_datamap_confidence(labels, scores, batch_idx, epoch, cfg)

        if VIS:
            _, preds = torch.max(scores.data, 1)
            total_t += labels.size(0)
            correct_t += (preds == labels).sum().item()

        if isinstance(criterion, nn.KLDivLoss):
            num_classes = scores.size(1)
            labels = torch.nn.functional.one_hot(labels, num_classes).type(torch.FloatTensor)
            with torch.cuda.device(device):
                labels = labels.cuda()

        if Unlearn_Clean:
            _, preds = torch.max(scores.data, 1)
            loss_inds = torch.eq(preds, labels).int()

            # with torch.no_grad():
            #     losses = criterion(scores, labels)
            #     losses = torch.maximum(losses, 0.0001*torch.ones_like(losses))
            #     inv_losses = 1/losses
            #     loss_weights = inv_losses / inv_losses.max()

            # all_losses = []
            # for cls in range(train_loader.dataset.num_classes):
            #     cls_labels = torch.ones_like(labels) * cls
            #     cls_inds = torch.not_equal(labels, cls_labels).int()
            #     cls_loss = criterion(scores, cls_labels) * cls_inds * loss_weights  # taking loss from non-gt class for correct predicted samples
            #     all_losses.append(cls_loss)
            #
            # target_loss = torch.cat(all_losses, dim=0) * Unlearn_Gamma

            # rand_labels = deepcopy(labels)
            # for i in range(len(labels)):
            #     with torch.cuda.device(device):
            #         all_cls = torch.arange(0, 10).cuda()
            #     non_gt_cls = torch.where(all_cls != labels[i])[0]
            #     perm_cls = non_gt_cls[torch.randperm(len(non_gt_cls))]
            #     rand_labels[i] = perm_cls[0]
            # target_loss = criterion(scores, rand_labels) * loss_inds
            # # target_loss = Unlearn_Gamma * target_loss
            # target_loss = Unlearn_Gamma * target_loss

            # if cycle == 1:
            #     target_loss = criterion(scores, labels)
            # else:
            #     target_loss = criterion(scores, labels) * loss_inds
            target_loss = criterion(scores, labels) * loss_inds
            # target_loss = criterion(scores, labels) * loss_weights
            # target_loss = criterion(scores, labels) * loss_weights * loss_inds
            # target_loss = EntropyLoss(reduction='none')(scores, labels)
            target_loss = -Unlearn_Gamma * target_loss
            # target_loss = -1/target_loss * Unlearn_Gamma
        else:
            target_loss = criterion(scores, labels)
            if LGA:
                target_loss = torch.sign(target_loss - cfgs.LGA_gamma) * target_loss

        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        loss            = m_backbone_loss

        loss.backward()

        # Clipping gradient for the stable training process
        # if warmup_train:
        #     clip_grad_norm_(models['backbone'].parameters(), max_norm=0.1)

        optimizers['backbone'].step()
        if method == 'lloss' or method == 'TA-VAAL':
            optimizers['module'].step()
    if VIS:
      #  print('total num in training {}'.format(total_t))
        return loss, 100 * correct_t / total_t
    else:
        return loss


def train_model(models, method, criterion, optimizers, schedulers, dataloaders, device, cfgs, cycle,
                log_data_map, NO_CLASSES, ref_models=None, warmup_train=False, first_train=False, ref_train=False,
                finetune=False):

    if warmup_train and ref_train:
        num_epochs = cfgs.BACKDOOR.warmup_epochs
        print(f">> Train a Model for WarmUp by {optimizers['backbone'].__class__.__name__} ...")

        ref_tf = dataloaders['train'].dataset.transform
        print('Train data transforms for reference model:', ref_tf)
    elif ref_train:
        if first_train:
            num_epochs = cfgs.BACKDOOR.ref_c1epochs
        else:
            num_epochs = cfgs.BACKDOOR.ref_epochs
        print(f">> Train Reference Model by {optimizers['backbone'].__class__.__name__} ...")
        ref_tf = dataloaders['train'].dataset.transform
        print('Train data transforms for reference model:', ref_tf)
    else:
        print(f">> Train Main Model by {optimizers['backbone'].__class__.__name__} ...")
        num_epochs = cfgs.OPTIMIZER_BACKBONE.EPOCH_TL

        main_tf = dataloaders['train'].dataset.transform
        print('Train data transforms for main model:', main_tf)

    print(f">> Labeled Loader {len(dataloaders['train'])}, Unlabeled Loader {len(dataloaders['train_unlabel'])}")

    epoch_loss = cfgs.OPTIMIZER_MODULE.EPOCHL
    MARGIN = cfgs.OPTIMIZER_MODULE.MARGIN
    WEIGHT = cfgs.OPTIMIZER_MODULE.WEIGHT
    best_acc = 0.

    rampup_length = cfgs.OPTIMIZER_BACKBONE.EPOCH_TL  # if finetune else cfgs.CYCLES
    semi_criterion = MixMatchLoss(rampup_length=rampup_length, lambda_u=cfgs.SSL.lambda_u)

    # if ref_train and cfgs.unlearn_ref and (not warmup_train):
    #     print('\nUnlearning on split clean set')
    #     # criterion_ul = EntropyLoss(reduction='none')
    #     criterion_ul = nn.CrossEntropyLoss(reduction='none')
    #     # ul_gamma = cfgs.unlearn_gamma * (cfgs.init_ratio * cfgs.NUM_TRAIN) / (len(dataloaders['train_clean']) * cfgs.BATCH + cfgs.BATCH)
    #     ul_gamma = cfgs.unlearn_gamma * len(dataloaders['train']) / len(dataloaders['train_clean'])
    #     print(f'Unlearning gamma = {ul_gamma:.6f}')
    #     for epoch in range(int(cfgs.unlearn_epochs)):
    #         loss_ul, acc_train_ul = train_epoch(models, method, criterion_ul, optimizers, schedulers, dataloaders, epoch,
    #                                             epoch_loss, device, MARGIN, WEIGHT, log_data_map, cfgs,
    #                                             warmup_train=warmup_train, VIS=True, Add_Noise=False,
    #                                             Unlearn_Clean=True, Unlearn_Gamma=ul_gamma)
    #
    #         acc_test_ul = test(models, method, dataloaders, device, mode='test')
    #         acc_test_bad_ul = test(models, method, dataloaders, device, mode='test_bad')
    #
    #         lr = optimizers['backbone'].param_groups[0]['lr']
    #         print(
    #             'epoch: {} \t lr: {:.4f} \t train_loss: {:.4f} \t train_acc: {:.2f} \t test_acc(bng/bad): {:.2f}/{:.2f}'.format(
    #                 'UL', lr, loss_ul, acc_train_ul, acc_test_ul, acc_test_bad_ul), flush=True)

    print('\nLearning on split poisoned set')
    for epoch in range(num_epochs):
        best_loss = torch.tensor([0.5]).cuda()
        # train_loss = calc_loss(models, dataloaders, criterion, device, 'train')
        # test_loss = calc_loss(models, dataloaders, criterion, device, 'test')
        if warmup_train or ref_train or cfgs.SSL.sl_train:
            if warmup_train or ref_train:
                LGA = cfgs.LGA
            else:
                LGA = False
            loss, acc_train = train_epoch(models, method, criterion, optimizers, schedulers, dataloaders, epoch,
                                          epoch_loss, device, MARGIN, WEIGHT, log_data_map, cfgs,
                                          warmup_train=warmup_train, VIS=True, Add_Noise=False,
                                          LGA=LGA
                                          )
        else:
            step = epoch  # if finetune else cycle
            loss = train_mixmatch_epoch(dataloaders, models['backbone'], optimizers['backbone'], semi_criterion,
                                        step, device, NO_CLASSES=NO_CLASSES)
            acc_train = test(models, method, dataloaders, device, mode='train')
        acc_test = test(models, method, dataloaders, device, mode='test')
        acc_test_bad = test(models, method, dataloaders, device, mode='test_bad')

        lr = optimizers['backbone'].param_groups[0]['lr']
        print('epoch: {} \t lr: {:.4f} \t train_loss: {:.4f} \t train_acc: {:.2f} \t test_acc(bng/bad): {:.2f}/{:.2f} \t'.format(epoch, lr, loss, acc_train, acc_test, acc_test_bad), flush=True)
        if 'backbone' in schedulers:
            schedulers['backbone'].step()
        if 'module' in schedulers and method in ('lloss', 'TA_VAAL'):
            schedulers['module'].step()
        if epoch % 50  == 0 and False:
            acc = test(models, method, dataloaders, device, mode='test')
            if best_acc < acc:
                best_acc = acc
                print('epoch: {} \t lr: {:.4f} \t Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(epoch, lr, acc, best_acc))

    if ref_train and cfgs.unlearn_ref and (not warmup_train):
        print('\nUnlearning on split clean set')
        # criterion_ul = EntropyLoss(reduction='none')
        criterion_ul = nn.CrossEntropyLoss(reduction='none')
        ul_gamma = cfgs.unlearn_gamma
        # ul_gamma = cfgs.unlearn_gamma * (cfgs.init_ratio * cfgs.NUM_TRAIN) / (len(dataloaders['train_clean']) * cfgs.BATCH + cfgs.BATCH)
        # ul_gamma = cfgs.unlearn_gamma * len(dataloaders['train']) / len(dataloaders['train_clean'])
        print(f'Unlearning gamma = {ul_gamma:.6f}')
        for epoch in range(int(cfgs.unlearn_epochs)):
            loss_ul, acc_train_ul = train_epoch(models, method, criterion_ul, optimizers, schedulers, dataloaders, epoch,
                                                epoch_loss, device, MARGIN, WEIGHT, log_data_map, cfgs,
                                                warmup_train=warmup_train, VIS=True, Add_Noise=False,
                                                Unlearn_Clean=True, Unlearn_Gamma=ul_gamma, cycle=cycle)

            acc_test_ul = test(models, method, dataloaders, device, mode='test')
            acc_test_bad_ul = test(models, method, dataloaders, device, mode='test_bad')

            lr = optimizers['backbone'].param_groups[0]['lr']
            print(
                'epoch: {} \t lr: {:.4f} \t train_loss: {:.4f} \t train_acc: {:.2f} \t test_acc(bng/bad): {:.2f}/{:.2f}'.format(
                    'UL', lr, loss_ul, acc_train_ul, acc_test_ul, acc_test_bad_ul), flush=True)

    print('\n>> Finished.')


def train_mixmatch_epoch(dataloaders, model, optimizer, criterion, step, device, NO_CLASSES, unlearn_reg=False,
                         alpha=0.75, T=0.5, fix_w=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    labeled_trainloader = dataloaders['train']
    unlabeled_trainloader = dataloaders['train_unlabel']

    # train_iteration = int(len(labeled_trainloader.sampler.indices) / labeled_trainloader.batch_size)*2
    train_iteration = 1024
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(train_iteration):

        try:
            inputs_x, targets_x = labeled_train_iter.next()
            if inputs_x.shape[0] != labeled_trainloader.batch_size:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            inputs_u, targets_poi = unlabeled_train_iter.next()
            if inputs_u.shape[0] != unlabeled_trainloader.batch_size:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, targets_poi = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, targets_poi = unlabeled_train_iter.next()
        inputs_u2 = deepcopy(inputs_u)

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, NO_CLASSES).scatter_(1, targets_x.view(-1,1).long(), 1)
        # targets_poi = torch.zeros(batch_size, 10).scatter_(1, targets_poi.view(-1, 1).long(), 1)

        with torch.cuda.device(device):
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()
            targets_poi = targets_poi.cuda()


        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        l = np.random.beta(alpha, alpha)

        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], step)

        # if not fix_w:
        #     num_labeled = len(labeled_trainloader)
        #     num_subset = len(unlabeled_trainloader)
        #     w = num_subset / num_labeled * w

        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('SSL-loss: {:.4f}({:.4f}, {:.4f}), W={:.2f}'.format(losses.avg, losses_x.avg, losses_u.avg, w))

    return losses.avg


def log_datamap_confidence(labels, scores, batch_idx, epoch, cfgs):
    # Find confidence for correct label
    confidence = torch.softmax(scores, dim=1)
    confidence_correct_label = [*map(lambda conf, label: conf[label].item(), confidence, labels)]

    # Compute dataset index for samples
    min_idx = batch_idx*cfgs.BATCH
    max_idx = min_idx + len(scores)
    indices = [*range(min_idx, max_idx)]

    # Load pandas dataframe
    results_path = os.path.join(cfgs.results_folder, cfgs.METHOD.replace('-', '_'))
    os.makedirs(results_path, exist_ok=True)
    results = os.path.join(results_path, cfgs.DATASET + '_full.csv')

    # Create a new dataframe if it doesn't exist or if we start a new training
    if not os.path.exists(results) or (epoch == 0 and 0 in indices):
        df = pd.DataFrame(index=[*range(cfgs.OPTIMIZER_BACKBONE.EPOCH_TL+1)],
                          columns=[*range(cfgs.NUM_TRAIN)])
    # Otherwise load it from a file
    else:
        df = pd.read_csv(results, index_col=0)

    # Set confidence values for current epoch and samples
    df.iloc[epoch+1, indices] = confidence_correct_label

    # Set the first row to be the real labels
    if epoch == 0:
        df.iloc[epoch, indices] = labels.tolist()

    df.to_csv(results)
