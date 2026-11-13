'''
    Anti-backdoor learning with HARVEY

    Note that: we borrow part of code from https://github.com/intellisec/eval-al and merge with backdoor defense of HARVEY
'''

# Python
import os
import random
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

import argparse
import datetime, psutil
import pickle
from copy import deepcopy

# Custom
from config import cfg_from_file
from train_test import train_model, test
from random_seed_generator import RandomSeedGenerator
from load_dataset import load_dataset
from util import Build_model, query_strategy, compute_poison_rate, test_tsne

from utils.loss import SCELoss


parser = argparse.ArgumentParser()
parser.add_argument("-c","--config_file",type=str, default='./conf/ResNet18_cifar10_SL_badnets_rate010.yml',
                    help="path of the config file")
parser.add_argument("-s", "--state", type=str, help="Recover state from pickled file")
parser.add_argument("-cuda","--cuda_visible_device", type=int, default=0, help="CUDA_VISIBLE_DEVICES")
parser.add_argument("-vis", "--tensorboard_vis", type=bool, default=False, help="enable visulization in tensorboard")
parser.add_argument("-dm", "--log_data_map", type=bool, default=False, help="log accuracies of each epoch for a datamap")
parser.add_argument("-det", "--deterministic", type=bool, default=False, help="train deterministic")

parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--ref_cycle", type=int, default=1, help="the cycle we retrieve the model for meta-splitting")

args = parser.parse_args()

# Main
if __name__ == '__main__':

    methods = ['SCE', 'sce', 'RCE', 'rce', 'CE', 'ce']
    datasets = ['cifar10', 'gtsrb', 'tinyimagenet']

    # Parse config file
    config_file = args.config_file
    cfgs = cfg_from_file(args.config_file)

    cfgs.RANDOM_SET_SEED = args.seed
    cfgs.RANDOM_MODEL_PARAM_SEED = args.seed

    print(cfgs)
    method = cfgs.second_stg
    ref_method = cfgs.ref_stg
    device = args.cuda_visible_device
    assert method in methods, 'No method %s! Try options %s'%(method, methods)
    assert cfgs.DATASET in datasets, 'No dataset %s! Try options %s'%(cfgs.DATASET, datasets)

    # base_dir = os.path.dirname(os.path.realpath(__file__)) 
    # results_folder = os.path.join(base_dir, cfgs.results_folder)   
    # print("results_folder: {}".format(results_folder))
    results_path = os.path.join(cfgs.results_folder, f"{cfgs.second_stg.replace('-', '_')}-{ref_method}")
    if not os.path.exists(results_path) :
        os.makedirs(results_path)
    print("results_path: {}".format(results_path))

    print("Dataset: %s"%cfgs.DATASET)
    print("Method type:%s"%cfgs.METHOD)
    print('Task learner:%s'%cfgs.TASK_LEARNER)
    print('Full set:', bool(cfgs.FULLSET))

    # Starting trial and cycle number
    trial0 = 0
    cycle0 = 0

    # Initialize the random seed generator for the initial set and generate seeds for each trial
    rand_set_seed_gen = RandomSeedGenerator(cfgs.RANDOM_SET_SEED)
    random_set_seeds = rand_set_seed_gen.get_k_random_numbers(cfgs.TRIALS)

    # Initialize the random seed generator for the model parameters and generate seeds for each trial
    rand_model_param_seed_gen = RandomSeedGenerator(cfgs.RANDOM_MODEL_PARAM_SEED)
    random_model_param_seeds = rand_model_param_seed_gen.get_k_random_numbers(cfgs.TRIALS)

    # Recover state from file
    recover = False
    recover_state = {}
    if args.state:
        recover = True
        with open(args.state, 'rb') as f :
            recover_state = pickle.load(f)
        # Overwrite starting trial and cycle
        trial0 = recover_state['trial']
        cycle0 = recover_state['cycle'] + 1
        random_set_seeds = recover_state['random_seed_states']
        random_model_param_seeds = recover_state['random_model_states']


    # Setup result file
    ex_setting = config_file.split('/')[-1].split('.')[0]
    write_mode = 'a' if recover else 'w'
    results = open(os.path.join(results_path, ex_setting+'.txt'), write_mode)

    # Check if full train 
    if cfgs.TOTAL_DATASET:
        print('Train TL once with all data points', flush=True)
        cfgs.TRIALS = 1
        cfgs.CYCLES = 1

    # Load training and testing dataset
    data_train, data_unlabeled, data_test_bad, data_test_clean, NO_CLASSES = load_dataset(cfgs.DATASET, cfgs)

    print('Train data transforms:', data_train.transform)
    print('Train unlabeld data transforms:', data_unlabeled.transform)
    print('Test bad data transforms:', data_test_bad.transform)
    print('Test clean data transforms:', data_test_clean.transform)

    indices_clean = list(data_train.clean_ids)
    indices_bad = list(data_train.poison_ids)
    indices_noise = list(data_train.noise_ids)

    # set seed for random init set selection
    if cfgs.RANDOM_SET_SEED:
        if cfgs.RANDOM_SET_SEED_ORDER:
            init_set_seed = random_set_seeds[cfgs.RANDOM_SET_SEED_ORDER]
        else:
            init_set_seed = random_set_seeds[0]
        random.seed(init_set_seed)
        print("Initial Random Set Seed:" + str(init_set_seed))

    '''
        Warm-up training for labeled_set initialization
    '''
    all_ids = list(range(len(data_train)))
    ref_dataloaders = {
        'train': DataLoader(data_unlabeled, batch_size=cfgs.BATCH, shuffle=True),  # sampler=SubsetRandomSampler(all_ids)),
        'train_clean': DataLoader(data_unlabeled, batch_size=cfgs.BATCH, shuffle=True),
        # 'train': DataLoader(data_train, batch_size=cfgs.BATCH, shuffle=True),  # sampler=SubsetRandomSampler(all_ids)),
        # 'train_clean': DataLoader(data_train, batch_size=cfgs.BATCH, shuffle=True),
        'train_unlabel': DataLoader(data_unlabeled, batch_size=cfgs.BATCH, shuffle=False),
        'train_tsne': DataLoader(data_unlabeled, batch_size=cfgs.BATCH, shuffle=False),
        'test': DataLoader(data_test_clean, batch_size=cfgs.BATCH),
        'test_bad': DataLoader(data_test_bad, batch_size=cfgs.BATCH)
    }

    start_time = time.asctime(time.localtime(time.time()))
    print("Start at: {}".format(start_time))

    # Load random seeds for the initial model parameters (returns None if no seed is configured)
    model_param_seed = random_model_param_seeds[0]
    ref_models = Build_model(data_unlabeled, method, NO_CLASSES, device, cfgs, args,
                             is_ref=True, model_param_seed=model_param_seed, recover=recover)
    recover = False

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Query strategy for ABL init
    ref_strategy = query_strategy(ref_method, ref_models, data_unlabeled, NO_CLASSES, cfgs, device, ref_model=True)
    ref_strategy.seed_random_init_set(init_set_seed)

    if cfgs.OPTIMIZER_BACKBONE.NAME_REF == 'SGD':
        optim_ref = optim.SGD(ref_models['backbone'].parameters(), lr=cfgs.OPTIMIZER_BACKBONE.LR_REF,
                              momentum=cfgs.OPTIMIZER_BACKBONE.MOMENTUM,
                              weight_decay=cfgs.OPTIMIZER_BACKBONE.WDECAY)
        # ref_sched_backbone = lr_scheduler.MultiStepLR(optim_ref, milestones=[cfgs.BACKDOOR.warmup_epochs])
        ref_sched_backbone = lr_scheduler.MultiStepLR(optim_ref, milestones=[50, 75])
        ref_schedulers = {'backbone': ref_sched_backbone}
    else:
        optim_ref = optim.Adam(ref_models['backbone'].parameters(), lr=cfgs.OPTIMIZER_BACKBONE.LR_REF)
        ref_schedulers = {}

    ref_optimizers = {'backbone': optim_ref}

    print(
        '\n'
        '>>>>>>>>>>>>>>>>>>>>>'
        ' Stage-1: Initialization of poisoned and benign sets '
        '<<<<<<<<<<<<<<<<<<<<<<'
        '\n'
    )

    # Train model on full train dataset
    print("Initial training on reference model with the entire poisoned dataset")
    criterion = nn.CrossEntropyLoss(reduction='none')
    # criterion = SCELoss(alpha=0.1, beta=1.0, reduce=False)
    # criterion = nn.KLDivLoss(reduction='none')
    train_model(ref_models, method, criterion, ref_optimizers, ref_schedulers,
                ref_dataloaders, device, cfgs, 0, args.log_data_map, NO_CLASSES,
                warmup_train=True, ref_train=True)

    ref_acc = test(ref_models, method, ref_dataloaders, device, mode='test')
    ref_acc_bad = test(ref_models, method, ref_dataloaders, device, mode='test_bad')

    if cfgs.plot_tsne:
        print("Plot hidden space distribution by T-SNE")
        test_tsne(cfgs, ref_dataloaders['train_tsne'], ref_models['backbone'], data_train.poison_ids, cycle=0)

    full_indices = list(range(len(data_train.dataset)))
    ref_strategy.feed_current_state(0, full_indices, [])

    ref_arg, num_clean, ref_best_th = ref_strategy.query()

    '''
        Lowest loss sampling + 
    '''
    print('Initialize benign set by half-and-half splitting')
    INIT_BUDGET = int(cfgs.init_ratio * len(data_train))
    UL_BUDGET = len(data_train) - INIT_BUDGET  # int(0.5*len(full_indices))
    labeled_set = list(torch.tensor(full_indices)[ref_arg][-UL_BUDGET:].numpy())

    subset = list(torch.tensor(full_indices)[ref_arg][:INIT_BUDGET].numpy())
    unlabeled_set = list(torch.tensor(full_indices)[ref_arg][INIT_BUDGET:-INIT_BUDGET].numpy())

    # Count immediate poison rate in labeled dataset
    poi_rate = compute_poison_rate(labeled_set, indices_bad, indices_clean, indices_noise)

    poi_rate_subset = compute_poison_rate(subset, indices_bad, indices_clean, indices_noise)

    num_poi_pool = int(len(labeled_set) * poi_rate)
    print(f'>> Benign dataset updated: {num_poi_pool}/{len(labeled_set)} ({poi_rate:.4f})')

    if cfgs.print_cls_dist:
        cls_cnt = np.zeros(NO_CLASSES)
        cnt_loader = DataLoader(data_unlabeled, batch_size=1, sampler=SubsetRandomSampler(labeled_set),
                                pin_memory=True)
        for i, (img, label) in enumerate(cnt_loader):
            label = label.squeeze().cpu().numpy()
            cls_cnt[label] += 1

        print(f"Class_wise distribution in split benign subset: \n{list(cls_cnt.astype('int'))}")

    print('Initial ulb pool {}'.format(len(unlabeled_set)))

    if cfgs.print_cls_dist:
        cls_cnt = np.zeros(NO_CLASSES)
        cnt_loader = DataLoader(data_unlabeled, batch_size=1, sampler=SubsetRandomSampler(subset),
                                pin_memory=True)
        for i, (img, label) in enumerate(cnt_loader):
            label = label.squeeze().cpu().numpy()
            cls_cnt[label] += 1

        print(f"Class_wise distribution in current subset: \n{list(cls_cnt.astype('int'))}")

    # print('Subset initialization by random sampling from ublabeled set')
    random.shuffle(unlabeled_set)

    print('Initial subset {}'.format(len(subset)))

    poi_rate_subset = compute_poison_rate(subset, indices_bad, indices_clean, indices_noise)

    train_loader = DataLoader(data_train,
                              batch_size=cfgs.BATCH,
                              sampler=SubsetRandomSampler(labeled_set),
                              pin_memory=True,
                              drop_last=False)
    train_unlabel_loader = DataLoader(data_train,
                                      batch_size=cfgs.BATCH,
                                      sampler=SubsetRandomSampler(subset+unlabeled_set),
                                      pin_memory=True,
                                      drop_last=False)
    train_tsne_loader = DataLoader(data_train, batch_size=cfgs.BATCH, shuffle=False)
    train_unlearn_loader = DataLoader(data_train,
                                      batch_size=cfgs.BATCH,
                                      sampler=SubsetRandomSampler(subset),
                                      pin_memory=True,
                                      drop_last=False)

    test_loader = DataLoader(data_test_clean, batch_size=cfgs.BATCH)
    test_loader_bad = DataLoader(data_test_bad, batch_size=cfgs.BATCH)
    dataloaders = {'train': train_loader,
                   'train_unlabel': train_unlabel_loader,
                   'train_tsne': train_tsne_loader,
                   'test': test_loader, 'test_bad': test_loader_bad}

    ref_dataloaders = {
        'train': DataLoader(data_unlabeled, batch_size=cfgs.BATCH, sampler=SubsetRandomSampler(subset), pin_memory=True, drop_last=False),
        'train_clean': DataLoader(data_unlabeled, batch_size=cfgs.BATCH, sampler=SubsetRandomSampler(labeled_set), pin_memory=True, drop_last=False),
        # 'train': DataLoader(data_train, batch_size=cfgs.BATCH, sampler=SubsetRandomSampler(subset), pin_memory=True, drop_last=False),
        # 'train_clean': DataLoader(data_train, batch_size=cfgs.BATCH, sampler=SubsetRandomSampler(labeled_set), pin_memory=True, drop_last=False),
        'train_unlabel': DataLoader(data_unlabeled, batch_size=cfgs.BATCH, sampler=SubsetRandomSampler(unlabeled_set), pin_memory=True, drop_last=False),
        'train_tsne': DataLoader(data_unlabeled, batch_size=cfgs.BATCH, shuffle=False),
        'test': DataLoader(data_test_clean, batch_size=cfgs.BATCH),
        'test_bad': DataLoader(data_test_bad, batch_size=cfgs.BATCH)
    }

    '''
        Prepare Main Model
    '''
    model_param_seed = random_model_param_seeds[0]
    print("Initial Random Model Seed:" + str(model_param_seed))
    models = Build_model(data_train, method, NO_CLASSES, device, cfgs, args,
                         model_param_seed=model_param_seed, recover=recover)
    recover = False

    second_strategy = query_strategy(method, models, data_train, NO_CLASSES, cfgs, device, is_main=True)
    second_strategy.seed_random_init_set(init_set_seed)

    if cfgs.OPTIMIZER_BACKBONE.NAME == 'SGD':
        optim_backbone = optim.SGD(models['backbone'].parameters(), lr=cfgs.OPTIMIZER_BACKBONE.LR,
                                   momentum=cfgs.OPTIMIZER_BACKBONE.MOMENTUM,
                                   weight_decay=1e-4)
                                   # weight_decay=cfgs.OPTIMIZER_BACKBONE.WDECAY)
        sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=cfgs.OPTIMIZER_BACKBONE.MILESTONES)
        schedulers = {'backbone': sched_backbone}
    else:
        optim_backbone = optim.Adam(models['backbone'].parameters(), lr=cfgs.OPTIMIZER_BACKBONE.LR)
        schedulers = {}
    optimizers = {'backbone': optim_backbone}

    num_clean_pre = 0
    num_clean = len(data_train)

    '''
        STAGE 2: isolating poisonous samples in progress
    '''
    print(
        '\n'
        '>>>>>>>>>>>>>>>>>>>>>>>>'
        ' Stage-2: Learning the backdoor in the progress '
        '<<<<<<<<<<<<<<<<<<<<<<<<'
        '\n'
    )
    for cycle in range(1, cfgs.CYCLES+1):

        # Measure cycle start time
        start_t = datetime.datetime.now()

        if cfgs.BACKDOOR.reinit_ref or cycle == 1:
            print("Re-initialize Reference Model")
            model_param_seed = random_model_param_seeds[0] if cfgs.RANDOM_MODEL_PARAM_SEED else None
            print("Initial Reference Model Seed:" + str(model_param_seed))
            ref_models = Build_model(data_unlabeled, method, NO_CLASSES, device, cfgs, args,
                                     is_ref=True, model_param_seed=model_param_seed, recover=recover)
            recover = False

            # Update strategy for re-initialized model
            ref_strategy = query_strategy(ref_method, ref_models, data_unlabeled, NO_CLASSES, cfgs, device, ref_model=True)
            ref_strategy.seed_random_init_set(init_set_seed)

            # ref_models['backbone'].classifier.reset_parameters()
            # print('Re-initialize model classifier')

        '''
            Training reference model with updated backdoor data portion
        '''

        if cycle > 1 or cfgs.BACKDOOR.ref_init_train:

            if cfgs.OPTIMIZER_BACKBONE.NAME_REF == 'SGD':
                optim_ref = optim.SGD(ref_models['backbone'].parameters(), lr=cfgs.OPTIMIZER_BACKBONE.LR_REF,
                                      momentum=cfgs.OPTIMIZER_BACKBONE.MOMENTUM,
                                      weight_decay=cfgs.OPTIMIZER_BACKBONE.WDECAY)
                ref_sched_backbone = lr_scheduler.MultiStepLR(optim_ref, milestones=[cfgs.BACKDOOR.ref_c1epochs+1])
                ref_schedulers = {'backbone': ref_sched_backbone}
            else:
                optim_ref = optim.Adam(ref_models['backbone'].parameters(), lr=cfgs.OPTIMIZER_BACKBONE.LR_REF)
                ref_schedulers = {}

            ref_optimizers = {'backbone': optim_ref}

            # Training and testing
            print('>> Start training reference model')
            print('>> cycle {}, num of samples in poisoned subset {}, poison ratio {:.4f} ({}/{})'.format(
                cycle-1, len(subset), poi_rate_subset, int(poi_rate_subset*len(subset)), len(subset)))

            first_train = True if cycle == 1 else False
            criterion = nn.CrossEntropyLoss(reduction='none')
            # alpha = max(0.1 - 0.01*cycle, 0.0)
            # criterion = SCELoss(alpha=alpha, beta=1.0, reduce=False)
            # criterion = nn.KLDivLoss(reduction='none')
            train_model(ref_models, method, criterion, ref_optimizers,
                        ref_schedulers, ref_dataloaders, device, cfgs, cycle, args.log_data_map,
                        NO_CLASSES, warmup_train=False, first_train=first_train, ref_train=True,
                        ref_models=None)

            if cfgs.plot_tsne and cycle in [1, cfgs.CYCLES]:
                print("Plot hidden space distribution by T-SNE")
                test_tsne(cfgs, ref_dataloaders['train_tsne'], ref_models['backbone'], data_train.poison_ids,
                          cycle=cycle)

            ref_acc = test(ref_models, method, ref_dataloaders, device, mode='test')
            ref_acc_bad = test(ref_models, method, ref_dataloaders, device, mode='test_bad')

        preserve_size = cfgs.BUDGET * 2
        preserved_subset = []

        num_dataset = len(data_train)
        len_subset = len(subset)
        # len_label = len(labeled_set) + BUDGET if cycle > 1 else INIT_BUDGET

        '''
            Run one iteration (cycle) of poisonous samples isolation
        '''
        print('>> First: Refresh Sampling Subset from the complete dataset')
        ref_test_set = list(range(len(data_unlabeled.dataset)))
        ref_strategy.feed_current_state(cycle, ref_test_set, labeled_set)
        ref_arg, num_clean, ref_best_th = ref_strategy.query()
        num_poison = len(data_train) - num_clean

        if cycle == args.ref_cycle:
            init_ref_models = deepcopy(ref_models)
            init_thresh = ref_best_th if cfgs.BACKDOOR.use_init_thresh else None
            init_strategy = query_strategy(ref_method, init_ref_models, data_unlabeled, NO_CLASSES, cfgs, device,
                                           best_th=init_thresh, ref_model=True, is_init=True)

        left_rate = cfgs.BACKDOOR.split_rate
        # len_subset = np.min((int(left_rate*num_poison), INIT_BUDGET))
        len_subset = np.max((int(left_rate*num_poison), int(cfgs.min_ratio * len(data_train))))
        # len_subset = num_poison

        pool_size = num_clean

        subset = list(torch.tensor(ref_test_set)[ref_arg][:len_subset].numpy())
        clean_pool = list(torch.tensor(ref_test_set)[ref_arg][-pool_size:].numpy())

        pool_poi_rate = compute_poison_rate(clean_pool, indices_bad, indices_clean, indices_noise)
        num_poi_pool = int(len(clean_pool) * pool_poi_rate)

        print(f'>> Benign dataset updated: {num_poi_pool}/{pool_size} ({pool_poi_rate:.4f})')

        if cfgs.cls_min_ratio > 0.0:
            # Counting the class distribution of current subset
            cls_cnt = np.zeros(NO_CLASSES)
            for id in subset:
                label = data_unlabeled.dataset[id][1]
                cls_cnt[label] += 1

            # Counting the number of samples for leaving in each class
            cls_cnt_ds = np.zeros(NO_CLASSES)
            for i in range(len(data_unlabeled.dataset)):
                label = data_unlabeled.dataset[i][1]
                cls_cnt_ds[label] += 1

            cls_cnt_ds = cls_cnt_ds * cfgs.cls_min_ratio
            cls_cnt_ds = cls_cnt_ds.astype(np.int32)

            # Append one sample into empty classes
            rest_set = list(torch.tensor(ref_test_set)[ref_arg][len_subset:].numpy())
            for id in rest_set:
                label = data_unlabeled.dataset[id][1]
                if cls_cnt[label] < cls_cnt_ds[label]:
                    subset.append(id)
                    cls_cnt[label] += 1

        cls_cnt = np.zeros(NO_CLASSES)
        cnt_loader = DataLoader(data_unlabeled, batch_size=1, sampler=SubsetRandomSampler(subset), pin_memory=True)
        for i, (img, label) in enumerate(cnt_loader):
            label = label.squeeze().cpu().numpy()
            cls_cnt[label] += 1

        if cfgs.print_cls_dist:
            print(f"Class_wise distribution in Subset: \n{list(cls_cnt.astype('int'))}")

        listd = list(torch.tensor(ref_test_set)[ref_arg][len_subset:-pool_size].numpy())

        len_label = pool_size
        print(">> Update label set to the full size of remaining samples with high loss value")

        labeled_set = clean_pool  # list(torch.tensor(clean_pool)[-len_label:].numpy())
        # pool_rest = list(torch.tensor(sample_pool)[:-len_label].numpy())
        #
        # listd = pool_rest + listd

        unlabeled_set = listd

        if cfgs.print_cls_dist:
            cls_cnt = np.zeros(NO_CLASSES)
            cnt_loader = DataLoader(data_train, batch_size=1, sampler=SubsetRandomSampler(labeled_set), pin_memory=True)
            for i, (img, label) in enumerate(cnt_loader):
                label = label.squeeze().cpu().numpy()
                cls_cnt[label] += 1

            print(f"Class_wise distribution in split benign set: \n{list(cls_cnt.astype('int'))}")

        # Construct subset for selection
        if not cfgs.TOTAL_DATASET:
            random.shuffle(unlabeled_set)
            unlabeled_set = preserved_subset + unlabeled_set  ### Starting from preserved subset

            # print('size of ulb pool {}'.format(len(unlabeled_set)))
            # print('size of subset {}'.format(len(subset)))

        # Count immediate poison rate in labeled dataset
        labeled_bad = []
        labeled_clean = []
        for i in labeled_set:
            if i in indices_bad:
                labeled_bad.append(i)
            else:
                assert i in indices_clean or i in indices_noise, f'Index {i} has no belonging!'
                labeled_clean.append(i)

        # Count immediate poison rate in labeled dataset
        poi_rate = compute_poison_rate(labeled_set, indices_bad, indices_clean, indices_noise)
        poi_rate_subset = compute_poison_rate(subset, indices_bad, indices_clean, indices_noise)

        # Create a new dataloader for the updated labeled dataset
        dataloaders['train'] = DataLoader(data_train, batch_size=cfgs.BATCH,
                                          sampler=SubsetRandomSampler(labeled_set),
                                          pin_memory=True)

        dataloaders['train_unlabel'] = DataLoader(data_train, batch_size=cfgs.BATCH,
                                                  # sampler=SubsetRandomSampler(subset + unlabeled_set),
                                                  sampler=SubsetRandomSampler(subset),
                                                  pin_memory=True)

        ref_dataloaders['train'] = DataLoader(data_unlabeled, batch_size=cfgs.BATCH,
                                              sampler=SubsetRandomSampler(subset),
                                              pin_memory=True)

        ref_dataloaders['train_clean'] = DataLoader(data_unlabeled, batch_size=cfgs.BATCH,
                                              sampler=SubsetRandomSampler(labeled_set),
                                              pin_memory=True)

        # ref_dataloaders['train'] = DataLoader(data_train, batch_size=cfgs.BATCH,
        #                                       sampler=SubsetRandomSampler(subset),
        #                                       pin_memory=True)
        #
        # ref_dataloaders['train_clean'] = DataLoader(data_train, batch_size=cfgs.BATCH,
        #                                             sampler=SubsetRandomSampler(labeled_set),
        #                                             pin_memory=True)

        ref_dataloaders['train_unlabel'] = DataLoader(data_unlabeled, batch_size=cfgs.BATCH,
                                                      sampler=SubsetRandomSampler(labeled_set),
                                                      pin_memory=True)

        #########################################################################################################
        #########################################################################################################

        '''
           Train Main Model with Updated De-Backdoored Label-Set 
        '''

        # Training and testing
        print('>> cycle {}, num of labeled data {}, poison rate(labeled/subset) {:.4f}({})/{:.4f}({})'.format(
            cycle, len(labeled_set),
            poi_rate, int(poi_rate*len(labeled_set)),
            poi_rate_subset, int(poi_rate_subset*len(subset)))
        )

        # if num_clean == num_clean_pre or num_clean >= int(0.5*len(data_train)):
        # if num_clean >= int(0.5 * len(data_train)):
        if num_clean == num_clean_pre:
            if cfgs.plot_tsne:
                # T-SNE plots on final reference model
                print("Plot hidden space distribution by T-SNE")
                test_tsne(cfgs, ref_dataloaders['train_tsne'], ref_models['backbone'], data_train.poison_ids, cycle)

            print(">> Early Stop.")
            break

        if cycle == cfgs.CYCLES:
            # Reached final training cycle
            print("Finished.")

            results.write('\n')
            break

        '''
            Update previous number of clean examples
        '''
        num_clean_pre = num_clean

        # Measure query time
        mem_cpu = psutil.virtual_memory().used / 1e9
        mem_gpu = torch.cuda.memory_allocated(device) / 1e9

        # Stop time
        end_t = datetime.datetime.now()
        took = end_t - start_t
        print(f'Cycle duration: {took}, allocated memory (cpu/gpu): {mem_cpu}GB / {mem_gpu}GB\n')
        results.write(f" '{took}' '{mem_cpu}' '{mem_gpu}'\n")
        # Ensure output is actually written to file
        results.flush()
        os.fsync(results.fileno())

    results.close()

    print(
        '\n'
        '>>>>>>>>>>>>>>>>>>>>'
        ' Stage-3: Meta splitting with the first reference model '
        '<<<<<<<<<<<<<<<<<<<<'
        '\n'
    )

    ref_test_set = list(range(len(data_unlabeled.dataset)))
    poison_pool = list(torch.tensor(ref_test_set)[ref_arg][:num_poison].numpy())
    poi_ratio_badset = compute_poison_rate(poison_pool, indices_bad, indices_clean, indices_noise)
    print(f"Feed {len(poison_pool)} (p-ratio={poi_ratio_badset:.4f}) isolated samples into Cycle-1 reference model")

    if not cfgs.BACKDOOR.meta_split:
        main_set = labeled_set
        rest_set = poison_pool
    else:
        init_strategy.feed_current_state(cycle, poison_pool, labeled_set)
        init_arg, init_num_clean, _ = init_strategy.query()
        init_num_poison = len(poison_pool) - init_num_clean
        main_set = labeled_set + list(torch.tensor(poison_pool)[init_arg][init_num_poison:].numpy())
        rest_set = list(torch.tensor(poison_pool)[init_arg][:init_num_poison].numpy())

    print(f"Size of benign set = {len(main_set)}, Size of Rest set with poison = {len(rest_set)}")

    dataloaders['train'] = DataLoader(data_train, batch_size=cfgs.BATCH,
                                      sampler=SubsetRandomSampler(main_set),
                                      pin_memory=True)

    dataloaders['train_unlabel'] = DataLoader(data_train, batch_size=cfgs.BATCH,
                                              sampler=SubsetRandomSampler(rest_set),
                                              pin_memory=True)

    cls_cnt = np.zeros(NO_CLASSES)
    cnt_loader = DataLoader(data_unlabeled, batch_size=1, sampler=SubsetRandomSampler(main_set),
                            pin_memory=True)

    if cfgs.print_cls_dist:
        for i, (img, label) in enumerate(cnt_loader):
            label = label.squeeze().cpu().numpy()
            cls_cnt[label] += 1

        print(f"Class_wise distribution in final benign-Set: \n{list(cls_cnt.astype('int'))}")

    main_poi_rate = compute_poison_rate(main_set, indices_bad, indices_clean, indices_noise)
    num_poi_main = int(len(main_set) * main_poi_rate)

    precision = 1 - num_poi_main/len(indices_bad)
    recall = (len(indices_bad) - num_poi_main) / len(main_set)

    print(
        f'>> Benign dataset updated: {num_poi_main}/{len(main_set)} ({main_poi_rate:.4f}), prec & recall [%] = {precision*100:.4f} {recall*100:.4f}')

    print(
        '\n'
        '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        ' Stage-4: Final clean model training '
        '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'
        '\n'
    )
    criterion = nn.CrossEntropyLoss(reduction='none')
    train_model(models, method, criterion, optimizers, schedulers, dataloaders, device, cfgs, 0,
                args.log_data_map, NO_CLASSES, warmup_train=False, ref_train=False, ref_models=None)

    # acc = test(models, method, dataloaders, device, mode='test')
    # acc_bad = test(models, method, dataloaders, device, mode='test_bad')
    #
    # print(
    #     'Cycle {}/{} || Label set size {}: Test acc (bng/bad) {:.2f}/{:.2f} | Ref Test acc (bng/bad) {:.2f}/{:.2f}'.format(
    #         cfgs.TRIALS, cycle + 1, cfgs.CYCLES, len(labeled_set), acc, acc_bad, ref_acc, ref_acc_bad
    #     ))

    if cfgs.plot_tsne:
        print("Plot hidden space distribution by T-SNE")
        test_tsne(cfgs, dataloaders['train_tsne'], models['backbone'], data_train.poison_ids, cycle, main_model=True)

    # print("Show the loss distribution on final trained main model")
    # second_strategy.feed_current_state(cycle, ref_test_set, labeled_set)
    # second_strategy.query()

    # Save current state
    with open(os.path.join(results_path, ex_setting + '_latest_state.pkl'), 'wb') as f:
        pickle.dump({
            'clean_set': main_set,
            'poison_set': rest_set,
            'random_seed_states': random_set_seeds,
            'random_model_states': random_model_param_seeds
        }, f, pickle.HIGHEST_PROTOCOL)

    # save current model parameters
    torch.save(models['backbone'].state_dict(), os.path.join(results_path, ex_setting + '_latest_target_model.pt'))

    end_time = time.asctime(time.localtime(time.time()))
    print("End at: {}".format(end_time))
