import sys
import os
import torch

from models.target_models import ResNet18, ResNet34
from models.wresnet import WRN161

from queries_strategies import *

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

activation = {}
ID = 0


def Build_model(data_train, method, NO_CLASSES, device, cfgs, args, is_ref=False, model_param_seed=None, recover=False):
    # Model - create new instance for every cycle so that it resets
    model_name = cfgs.REF_LEARNER if is_ref else cfgs.TASK_LEARNER
    if model_name=="resnet18":
        model    = ResNet18(num_classes=NO_CLASSES, fixed_model_parameter_seed=model_param_seed).to(device)
    elif model_name=="resnet34":
        model    = ResNet34(num_classes=NO_CLASSES, fixed_model_parameter_seed=model_param_seed).to(device)
    elif model_name=="wrn161":
        model = WRN161(num_classes=NO_CLASSES, fixed_model_parameter_seed=model_param_seed).to(device)
        torch.manual_seed(model_param_seed)
    else:
        print('choose a valid acquisition function', flush=True)
        raise ValueError

    print(f'build model {model_name}')
    if recover:
        backbone_state = args.state.split('state')[0]+'target_model.pt'
        print('recover target model from the pickled file : {} '.format(backbone_state))
        model.load_state_dict(torch.load(backbone_state, map_location=device))
        if method in ('lloss', 'TA-VAAL'):
            lossmodule_state = args.state.split('state')[0]+'loss_model.pt'
            print('recover loss module from the pickled file : {}'.format(lossmodule_state))
            loss_module.load_state_dict(torch.load(lossmodule_state, map_location=device))

    models      = {'backbone': model}
    if method in ('lloss', 'TA-VAAL'):
        models = {'backbone': model, 'module': loss_module}
    return models

def query_strategy(method, model, data_unlabeled, NO_CLASSES, cfgs, device, best_th=None,
                   ref_model=False, is_main=False, is_init=False, run_dir=None):
    if method == 'SCE' or method == 'sce':
        strategy = SCEntropy(model, data_unlabeled, NO_CLASSES, cfgs, device,
                             best_th=best_th, ref_model=ref_model, is_main=is_main, is_init=is_init,
                             alpha=0.1, beta=1.0, run_dir=run_dir)
    elif method == 'RCE' or method == 'rce':
        strategy = SCEntropy(model, data_unlabeled, NO_CLASSES, cfgs, device,
                             best_th=best_th, ref_model=ref_model, is_main=is_main, is_init=is_init,
                             alpha=0.0, beta=1.0, run_dir=run_dir)
    elif method == 'CE' or method == 'ce':
        strategy = SCEntropy(model, data_unlabeled, NO_CLASSES, cfgs, device,
                             best_th=best_th, ref_model=ref_model, is_main=is_main, is_init=is_init,
                             alpha=1.0, beta=0.0, run_dir=run_dir)
    else:
        print('choose a valid querying function', flush=True)
        raise ValueError
    return strategy

def display_progress(text, current_step, last_step, enabled=True,
                     fix_zero_start=True):
    """Draws a progress indicator on the screen with the text preceeding the
    progress
    Arguments:
        test: str, text displayed to describe the task being executed
        current_step: int, current step of the iteration
        last_step: int, last possible step of the iteration
        enabled: bool, if false this function will not execute. This is
            for running silently without stdout output.
        fix_zero_start: bool, if true adds 1 to each current step so that the
            display starts at 1 instead of 0, which it would for most loops
            otherwise.
    """
    if not enabled:
        return

    # Fix display for most loops which start with 0, otherwise looks weird
    if fix_zero_start:
        current_step = current_step + 1

    term_line_len = 80
    final_chars = [':', ';', ' ', '.', ',']
    if text[-1:] not in final_chars:
        text = text + ' '
    if len(text) < term_line_len:
        bar_len = term_line_len - (len(text)
                                   + len(str(current_step))
                                   + len(str(last_step))
                                   + len("  / "))
    else:
        bar_len = 30
    filled_len = int(round(bar_len * current_step / float(last_step)))
    bar = '=' * filled_len + '.' * (bar_len - filled_len)

    bar = f"{text}[{bar:s}] {current_step:d} / {last_step:d}"
    if current_step < last_step-1:
        # Erase to end of line and print
        sys.stdout.write("\033[K" + bar + "\r")
    else:
        sys.stdout.write(bar + "\n")

    sys.stdout.flush()


def compute_poison_rate(target_set, indices_bad, indices_clean, indices_noise):
    collect_bad = []
    collect_clean = []
    for i in target_set:
        if i in indices_bad:
            collect_bad.append(i)
        else:
            assert i in indices_clean or i in indices_noise, f'Index {i} has no belonging!'
            collect_clean.append(i)

    poi_rate = len(collect_bad) / len(target_set)

    return poi_rate


def get_activation(name):
    def hook(model, input, output):
        # output = F.avg_pool2d(output, 8)
        output = output.flatten(1,3).detach().cpu().numpy()
        # output = output.detach().cpu().numpy()
        global ID
        activation[name+'_'+str(ID)] = output
        ID += 1
    return hook


def test_tsne(cfgs, test_loader, model_ascent, poison_idx, cycle, main_model=False, device=None, run_dir=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    if run_dir is None:
        run_dir = os.path.join("outputs", cfgs.BACKDOOR.trigger_type.replace('Trigger', '').replace('Pixel', ''), cfgs.DATASET, cfgs.ref_stg + "_tSNE")

    global activation
    global ID

    activation = {}
    ID = 0

    opt = cfgs.BACKDOOR
    ds_name = cfgs.DATASET
    model_name = cfgs.TASK_LEARNER
    trigger_name = opt.trigger_type.replace('Trigger', '').replace('Pixel', '')

    model_ascent.eval()

    all_target = np.array([])

    for idx, (img, target) in enumerate(test_loader, start=1):

        all_target = np.append(all_target, target)

        img = img.to(device)
        target = target.to(device)

        with torch.no_grad():

            # Add hook on output of last residual block
            l_hook = model_ascent.avgpool.register_forward_hook(get_activation(f'last_block_output_b{idx}'))
            output = model_ascent(img)
            l_hook.remove()

            # batch_losses = torch.nn.CrossEntropyLoss(reduce=False)(output, target).detach().cpu().numpy()

    activations = None
    for k in activation.keys():
        # act_vector = np.expand_dims(activation[k].flatten(), 0)
        act_vector = activation[k]
        if activations is None:
            activations = act_vector
        else:
            activations = np.concatenate((activations, act_vector), axis=0)
        # print(f'{k} is added')

    print(f"Collected activations with shape {activations.shape}")

    # for dim in range(activations.shape[-1]):
    #     dim_mean = sum(activations[:, dim]) / len(activations)
    #     dim_std = (sum((activations[:, dim] - dim_mean) ** 2) / len(activations)) ** (1 / 2.0)
    #
    #     activations[:, dim] = (activations[:, dim] - dim_mean) / dim_std

    add_name = 'mainFinal' if main_model else ''
    dir_name = os.path.join(run_dir, "tsne")
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    act_fname = os.path.join(dir_name, f'{model_name}_tSNE_{opt.inject_portion}_{cfgs.INIT_BUDGET}_c{cycle}_{add_name}.npy')
    np.save(act_fname, activations)

    # tsne_results = TSNE(n_iter=1000, perplexity=100, num_neighbors=200).fit_transform(activations)
    tsne_results = TSNE(n_iter=500, perplexity=10, n_components=2).fit_transform(activations)
    print(f"t-SNE projection with shape {tsne_results.shape}")

    add_name = f"{opt.trigger_type}"

    # Create the scatter
    cdict = {0: '#17becf', 1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c', 4: '#d62728',
             5: '#9467bd', 6: '#8c564b', 7: '#e377c2', 8: '#7f7f7f', 9: '#bcbd22'}

    cdict.update({opt.target_label: '#3b3b3b'})

    '''
        Plot original t-SNE plots
    '''
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, title='T-SNE (' + add_name + ')')

    # plot distribution of all data
    for label in np.unique(all_target):
        idx = np.where(all_target == label)
        ax.scatter(
            x=tsne_results[idx, 0],
            y=tsne_results[idx, 1],
            c=cdict[label],
            label=f"Class: {int(label)}",
            cmap=plt.cm.get_cmap('Paired'),
            alpha=0.6,
            s=1)

    # plot emphasized points on poisoned data
    ax.scatter(
        x=tsne_results[poison_idx, 0],
        y=tsne_results[poison_idx, 1],
        c='black',
        label="Poisoned",  # f"Class: {opt.target_label} (target)",
        cmap=plt.cm.get_cmap('Paired'),
        marker='o',
        s=4)

    add_name = 'mainFinal' if main_model else ''
    plt.savefig(os.path.join(dir_name, f'{model_name}_tSNE_{opt.inject_portion}_{cfgs.INIT_BUDGET}_c{cycle}_{add_name}.pdf'))
    plt.savefig(os.path.join(dir_name, f'{model_name}_tSNE_{opt.inject_portion}_{cfgs.INIT_BUDGET}_c{cycle}_{add_name}.png'))
    plt.close(fig)