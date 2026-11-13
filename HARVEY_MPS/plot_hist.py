import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import argparse

from math import pi

cdict = {
    "primary": "#006BA4",
    "secondary": "#ff800e",
    "tertiary": "#2ca02c"
}

ticks_size = "large"
legend_size = "large"
title_size = "large"
label_size = "x-large"


def add_subplot_axes(ax, probs_poi, probs_cln, bins, fname_pre):
    probs_poi = np.load(f"{fname_pre}_c{cycle}_ref_init_probs_poi.npy")
    probs_cln = np.load(f"{fname_pre}_c{cycle}_ref_init_probs_cln.npy")
    best_th = np.load(f"{fname_pre}_c{cycle}_ref_init_otsu_th.npy")

    _, bins = np.histogram(np.hstack((probs_poi, probs_cln)), bins=70)

    rect = [0.18, 0.22, 0.6, 0.65]

    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    #subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+

    xlims = [-0.2, 12]  # if not is_init else [-0.5, 12]
    ylims = [0, 5000]

    subax = fig.add_axes([x,y,width,height])
    subax.hist(probs_cln, bins=bins, color=cdict['primary'], alpha=1.0, label='Benign samples')
    subax.hist(probs_poi, bins=bins, color=cdict['secondary'], alpha=0.8, label='Poisonous samples')

    subax.plot([best_th, best_th], [0, ylims[1]], color='black', linestyle='dashed', linewidth=2.5, label="OTSU threshold")

    xticks = [i for i in range(xlims[1] + 1)]
    subax.set_xticks(xticks, labels=xticks, fontsize=ticks_size)
    subax.set_xlim(xlims[0], xlims[1])

    # if not is_init else [0, 10000]
    yticks = [i * 1000 for i in range(int(ylims[1] / 1000) + 1)]
    yticklabels = [i for i in range(int(ylims[1] / 1000) + 1)]
    subax.set_yticks(yticks, labels=yticklabels, fontsize=ticks_size)
    subax.set_ylim(ylims[0], ylims[1])

    subax.set_title("Meta-Splitting", fontsize=title_size)

    subax.annotate("$x 10^3$", (-0.2, int(ylims[1] * 1.02)), annotation_clip=False, fontsize=ticks_size)

    subax.legend(fontsize=legend_size)

    return subax


def add_zoomin_subplot(ax, probs_poi, probs_cln, bins, axisbg='w'):
    rect = [0.4, 0.2, 0.4, 0.4]

    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    #subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x,y,width,height])
    subax.hist(probs_cln, bins=bins, color=cdict['primary'], alpha=1.0, label='Benign samples')
    subax.hist(probs_poi, bins=bins, color=cdict['secondary'], alpha=0.8, label='Poisonous samples')

    xlims = [8, 11]  # if not is_init else [-0.5, 12]
    xticks = [i for i in range(xlims[1] + 1)]
    subax.set_xticks(xticks, labels=xticks, fontsize=ticks_size)
    subax.set_xlim(xlims[0], xlims[1])

    ylims = [0, 800]  # if not is_init else [0, 10000]
    yticks = [i * 200 for i in range(int(ylims[1] / 200) + 1)]
    yticklabels = [i*2 for i in range(int(ylims[1] / 200) + 1)]
    subax.set_yticks(yticks, labels=yticklabels, fontsize=ticks_size)
    subax.set_ylim(ylims[0], ylims[1])

    subax.annotate("$x 10^2$", (xlims[0], int(ylims[1] * 1.02)), annotation_clip=False, fontsize=ticks_size)

    return subax


def plot_histogram(trigger_type, dataset, ref_stg, task_learner, target_label, inject_portion, optim_ref, init_budget,
              cycle, ref_epochs, plot_format, trainer, num_bins, add_name, is_init=True, is_main=False, add_legend=False):
    fname_pre = f"outputs/{trigger_type.replace('Trigger', '').replace('Pixel', '')}/{dataset}/{ref_stg}/{trainer}/{task_learner}_" \
                f"T{target_label}_prate{str(inject_portion).replace('.', '')}_{optim_ref}_{init_budget}_{ref_epochs}"

    if is_main:
        model_name = "main"
    elif is_init:
        model_name = "ref_init"
    else:
        model_name = "ref"

    probs = np.load(f"{fname_pre}_c{cycle}_{model_name}_probs.npy")
    probs_poi = np.load(f"{fname_pre}_c{cycle}_{model_name}_probs_poi.npy")
    probs_cln = np.load(f"{fname_pre}_c{cycle}_{model_name}_probs_cln.npy")
    best_th = np.load(f"{fname_pre}_c{cycle}_{model_name}_otsu_th.npy")

    outs, bins = np.histogram(np.hstack((probs_poi, probs_cln)), bins=num_bins)

    poi_outs, poi_bins = np.histogram(probs_poi, bins=bins)
    cln_outs, cln_bins = np.histogram(probs_cln, bins=bins)

    print(f'\nbins\tnum_cln')
    for i in range(len(bins[1:])):
        num_bins = bins[i]
        num_cln = cln_outs[i] / 1000
        print(f'{num_bins}\t{num_cln}')

    print(f'\nbins\tnum_p')
    for i in range(len(bins[1:])):
        num_bins = bins[i]
        num_poi = poi_outs[i] / 1000
        if num_poi > 0.001:
            print(f'{num_bins}\t{num_poi}')

    print(f'\nOtsu threshold: {best_th:.4f}')

    fig = plt.figure(figsize=(4.5, 3))
    ax = fig.add_axes((0.13, 0.16, 0.84, 0.77))
    # ax.hist(probs, bins=bins, color="black", alpha=0.3)

    ax.hist(probs_cln, bins=bins, color=cdict['primary'], alpha=1.0, label='Benign samples')
    ax.hist(probs_poi, bins=bins, color=cdict['secondary'], alpha=0.8, label='Poisonous samples')


    xlims = [-0.5, 12]  # if not is_init else [-0.5, 12]
    xticks = [i for i in range(xlims[1]+1)]
    ax.set_xticks(xticks, labels=xticks, fontsize=ticks_size)
    ax.set_xlim(xlims[0], xlims[1])

    ylims = [0, 10000]  # if not is_init else [0, 10000]
    yticks = [i*1000 for i in range(int(ylims[1]/1000)+1)]
    yticklabels = [i for i in range(int(ylims[1]/1000)+1)]
    ax.set_yticks(yticks, labels=yticklabels, fontsize=ticks_size)
    # ax.set_yscale('log')
    # ax.set_ylim(0, 10000)
    ax.set_ylim(ylims[0], ylims[1])

    ax.annotate("$x 10^3$", (-0.5, int(ylims[1]*1.02)), annotation_clip=False, fontsize=ticks_size)

    # ax.annotate("", xy=(10.5, 3000), xycoords='data', xytext=(11, 700), textcoords='data',
    #             arrowprops=dict(arrowstyle="->", connectionstyle="angle3, angleA=40, angleB=135", lw=2, mutation_scale=20),
    #             )
    # add_zoomin_subplot(ax, probs_poi, probs_cln, bins, fname_pre)

    # plt.rc('font', size=15)
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,0))
    # ax.yaxis.major.formatter._useMathText = True
    # ax.yaxis.get_offset_text().set_fontsize(15)
    # ax.set_yticks(yticks, labels=yticklabels, fontsize=15)

    # ax.grid(linestyle='--', color='gray')

    # ylims = ax.get_ylim()

    if cycle > 0 and not is_main:
        ax.plot([best_th, best_th], [0, ylims[1]], color='black', linestyle='dashed', linewidth=2.5, label="OTSU threshold")

    if add_legend:
        ax.legend(fontsize=legend_size)  #, loc="upper left")

    # ax.set_title(f"SCEntropy Distribution (data {len(probs)})")

    ax.set_xlabel("CE Loss", fontsize=label_size)
    ax.set_ylabel("Number of Samples", fontsize=label_size)

    add_fname = f"_{add_name}" if add_name != "" else ""
    fname_out = f"./plots/{trigger_type.replace('Trigger', '').replace('Pixel', '')}_{task_learner}_T{target_label}_" \
                f"prate{str(inject_portion).replace('.', '')}_{optim_ref}_{init_budget}_Hist_{model_name}_c{cycle}{add_fname}.{plot_format}"

    plt.savefig(fname_out)
    plt.close(fig)

    print(f"Save data histogram plot for cycle{cycle}")

    return best_th


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", type=int, default=0, help="HARVEY cycle for plotting")
    parser.add_argument("--is_init", action='store_true', default=False, help="Print initial model for meta-splitting")
    parser.add_argument("--is_main", action='store_true', default=False, help="Print main model")
    parser.add_argument("--add_legend", action='store_true', default=False, help="Add legend")
    parser.add_argument("--format", type=str, default='png', choices=['png', 'pdf'], help="Format of output plot")
    parser.add_argument("--trainer", type=str, default='SL', choices=['SL', 'SSL'], help="Trainer")
    parser.add_argument("--bins", type=int, default=200, help="Number of bins for plotting")
    parser.add_argument("--ref_stg", type=str, default='RCE_0.0_1.0', help="loss function for splitting")
    parser.add_argument("--add_name", type=str, default='', help="Trainer")

    args = parser.parse_args()

    trigger_type = "blend"
    dataset = "cifar10"
    task_learner = "resnet18"
    target_label = 0
    inject_portion = 0.1
    optim_ref = "ADAM"
    init_budget = 20000
    ref_epochs = 10
    cycle = args.cycle

    plot_histogram(trigger_type, dataset, args.ref_stg, task_learner, target_label, inject_portion, optim_ref, init_budget,
                   cycle, ref_epochs, num_bins=args.bins, add_name=args.add_name, plot_format=args.format, trainer=args.trainer,
                   is_init=args.is_init, is_main=args.is_main, add_legend=args.add_legend)
