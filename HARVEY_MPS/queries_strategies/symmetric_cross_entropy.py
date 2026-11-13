import os
import torch
import numpy as np
from utils.loss import SCELoss
from torch.utils.data import DataLoader
#custom
from .strategy import Strategy
from data.sampler import SubsetSequentialSampler

import matplotlib.pyplot as plt


def otsu_threshold(input, lbound=1.0, ubound=90.0, max_loss=15.0, norm_loss=False):
    # assert img.ndim == 2, "input should be a 2D image"

    input = np.array(input) if type(input).__name__ == 'list' else input

    max_input = np.max(input)

    if norm_loss:
        input = input/max_input

    th_step = 0.1  # 1.0 / len(input)

    th_begin = np.percentile(input, lbound)  # lbound/100.0*max_input
    th_end = np.percentile(input, ubound)  # ubound/100.0*max_input

    print(f"Otsu threhold range: [{th_begin:.4f}, {th_end:.4f}]")

    # th_begin = np.min(np.max(input), max_loss)*lbound/100.0
    # th_end = np.min(np.max(input), max_loss)*ubound/100.0

    # th_begin = lbound/100.0
    # th_end = ubound/100.0

    max_g = 0
    best_thresh = th_begin

    for thresh in np.arange(th_begin, th_end, th_step):
        bin_img = input > thresh
        bin_img_inv = input <= thresh

        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        # if 0 == fore_pix:
        #     continue
        # if 0 == back_pix:
        #     break

        w0 = float(fore_pix) / input.size
        u0 = float(np.sum(input * bin_img)) / fore_pix
        w1 = float(back_pix) / input.size
        u1 = float(np.sum(input * bin_img_inv)) / back_pix

        g = w0 * w1 * (u0 - u1) * (u0 - u1)

        if g >= max_g:
            max_g = g
            best_thresh = thresh

    output_thresh = best_thresh * max_input if norm_loss else best_thresh

    print(f"Found threshold: {output_thresh:.4f}")

    return output_thresh


class SCEntropy(Strategy):
    def __init__(self, model, data_unlabeled, NO_CLASSES, cfgs, device, best_th=None, ref_model=False, is_main=False, is_init=False, alpha=0.1, beta=1.0, run_dir=None):
        super(SCEntropy, self).__init__(model, data_unlabeled, NO_CLASSES, cfgs, device)

        self.cfgs = cfgs
        self.NO_CLASSES = NO_CLASSES
        self.ref_model = ref_model

        self.is_main = is_main
        self.is_init = is_init
        self.best_th = best_th

        self.trainer = "SL" if cfgs.SSL.sl_train else "SSL"

        self.alpha = alpha
        self.beta = beta
        self.run_dir = run_dir

    def plot_hist(self, probs, poi_ids, cln_ids, best_th, labels, dir_name, is_ref=False):

        NUM_BINS = 200

        xlims = [-1, 15]
        ylims = [0, 10000] if not self.is_init else [0, 5000]

        probs_poi = probs[poi_ids]
        probs_cln = probs[cln_ids]
        bins = np.histogram(np.hstack((probs_poi, probs_cln)), bins=NUM_BINS)[1]

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        # ax.hist(probs, bins=bins, color="black", alpha=0.3)
        ax.hist(probs_cln, bins=bins, color="cyan", alpha=0.8)
        ax.hist(probs_poi, bins=bins, color="red", alpha=0.9)

        # ylims = ax.get_ylim()
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])
        if self.cycle > 0 and not self.is_main:
            ax.plot([best_th, best_th], [0, ylims[1]], color='black', linestyle='dashed', linewidth=2.5)

        ax.set_title(f"SCEntropy Distribution (data {len(probs)})")

        fname_pre = f"{dir_name}/{self.cfgs.REF_LEARNER}_T{self.cfgs.BACKDOOR.target_label}_" \
                    f"prate{str(self.cfgs.BACKDOOR.inject_portion).replace('.','')}_" \
                    f"{self.cfgs.OPTIMIZER_BACKBONE.NAME_REF}_{self.cfgs.INIT_BUDGET}_{self.cfgs.BACKDOOR.ref_epochs}"

        if self.is_main:
            model_name = "main"
        elif self.is_init:
            model_name = "ref_init"
        else:
            model_name = "ref"

        np.save(f"{fname_pre}_c{self.cycle}_{model_name}_probs.npy", probs)
        np.save(f"{fname_pre}_c{self.cycle}_{model_name}_probs_poi.npy", probs_poi)
        np.save(f"{fname_pre}_c{self.cycle}_{model_name}_probs_cln.npy", probs_cln)
        np.save(f"{fname_pre}_c{self.cycle}_{model_name}_otsu_th.npy", best_th)

        if self.cycle == 0:
            np.save(f"{fname_pre}_c{self.cycle}_{model_name}_labels.npy", labels)

        fname = f"{fname_pre}_SCEhist_otsu_{model_name}_c{self.cycle}.png"
        # if is_ref:
        #     plt.savefig(f"./{self.cfgs.DATASET}_{self.cfgs.second_stg}_SCEntropy_histogram_main.png")
        # else:
        plt.savefig(fname)
        plt.close(fig)

        print(f"Save data histogram plot for cycle{self.cycle}")

        return best_th

    def plot_prate(self, probs, poi_ids, cln_ids, dir_name):

        prob_args = np.argsort(probs)

        data_step = int(len(probs)*0.02)
        num_iters = int(len(probs) / data_step)
        num_batch = data_step
        prate_list = []
        pnum_list = []
        data_list = []
        for i in range(num_iters):
            batch_args = prob_args[:num_batch]
            bad_list = [id for id in batch_args if id in poi_ids]
            p_num = len(bad_list)
            p_rate = p_num / num_batch

            data_list.append(num_batch)
            pnum_list.append(p_num)
            prate_list.append(p_rate)

            num_batch += data_step

        data_list = np.array(data_list)
        pnum_list = np.array(pnum_list)
        prate_list = np.round(np.array(prate_list)*100.0, 2)

        fig = plt.figure(figsize=(12, 6))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(data_list, pnum_list, color="black", marker='o', markersize=10, linewidth=1.5)
        ax1.set_ylim(0, 5500)
        ax1.set_xlabel("N Examples")
        ax1.set_ylabel("N Poisoned Examples")
        ax1.set_title(f"Number of Poisoned Examples in Ascent SCE loss")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(data_list, prate_list, color="red", marker='o', markersize=10, linewidth=1.5)
        ax2.set_ylim(0, 100)
        ax2.set_xlabel("N Examples")
        ax2.set_ylabel("Poisoning Rate [%]")
        ax2.set_title(f"Poisoning Rate in Ascent SCE loss")

        fname_pre = f"{dir_name}/{self.cfgs.REF_LEARNER}_T{self.cfgs.BACKDOOR.target_label}_" \
                    f"prate{str(self.cfgs.BACKDOOR.inject_portion).replace('.','')}_" \
                    f"{self.cfgs.OPTIMIZER_BACKBONE.NAME_REF}_{self.cfgs.INIT_BUDGET}"
        model_name = "main" if self.is_main else "ref"

        np.save(f"{fname_pre}_PnumList_SCE_{model_name}_c{self.cycle}.npy", np.asarray(pnum_list))
        np.save(f"{fname_pre}_PrateList_SCE_{model_name}_c{self.cycle}.npy", np.asarray(prate_list))

        fname = f"{fname_pre}_DataDist_SCE_{model_name}_c{self.cycle}.png"
        # if is_ref:
        #     plt.savefig(f"./{self.cfgs.DATASET}_{self.cfgs.second_stg}_SCEntropy_histogram_main.png")
        # else:
        plt.savefig(fname)
        plt.close(fig)

        print(f"Save p_rate plots for cycle{self.cycle}")

        return best_th

    def query(self):
        pin_mem = self.device.type == 'cuda'
        unlabeled_loader = DataLoader(self.data_unlabeled, batch_size=self.BATCH,
                                    sampler=SubsetSequentialSampler(self.subset),
                                    pin_memory=pin_mem)

        losses, labels = self.get_predict_prob(unlabeled_loader)
        U = losses.cpu().numpy()
        label_all = labels.cpu().numpy()

        arg = np.argsort(U)

        # if self.ref_model and self.cfgs.BACKDOOR.refresh_subset:
        indices_poi = self.data_unlabeled.poison_ids
        indices_cln = self.data_unlabeled.clean_ids

        # Indexes only existing in subset list
        inds_poi = [i for i, id in enumerate(self.subset) if id in indices_poi]
        inds_cln = [i for i, id in enumerate(self.subset) if id in indices_cln]

        # best_th = 10.0
        if self.best_th is None:
            if self.alpha == 0.0:
                best_th = 4.6052  # 0.5 * (-log(1e-4))
                # perc = 48.0
                # best_th = np.percentile(U, 100 - perc)
            else:
                # best_th = otsu_threshold(U)
                best_th = 5.0
        else:
            best_th = self.best_th
            print(f"In loss range ({U.min():.4f}, {U.max():.4f}), threshold value = {best_th:.4f}")

        num_clean = np.sum(U >= best_th)

        if self.cfgs.plot_hist:
            # if self.cycle == 0 or self.cycle == 1:
            if self.run_dir is not None:
                dir_name = os.path.join(self.run_dir, "rce_histograms")
            else:
                dir_name = f"outputs/{self.cfgs.BACKDOOR.trigger_type.replace('Trigger', '').replace('Pixel', '')}/" \
                           f"{self.cfgs.DATASET}/{self.cfgs.ref_stg}_{self.alpha}_{self.beta}/{self.trainer}"
            if not os.path.isdir(dir_name):
                os.makedirs(dir_name)

            self.plot_prate(U, inds_poi, indices_cln, dir_name=dir_name)
            self.plot_hist(U, inds_poi, inds_cln, best_th, label_all, dir_name=dir_name)

        return arg, num_clean, best_th
        # else:
        #     return arg

    def get_predict_prob(self, unlabeled_loader):

        self.model['backbone'].eval()
        loss_all = torch.tensor([], device=self.device)
        labels = torch.tensor([], device=self.device)

        with torch.no_grad():
            for inputs, targets in unlabeled_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                predict = self.model['backbone'](inputs)
                losses = SCELoss(alpha=self.alpha, beta=self.beta, num_classes=self.NO_CLASSES, reduce=False)(predict, target=targets)
                # losses = torch.nn.CrossEntropyLoss(reduction='none')(predict, targets)
                loss_all = torch.cat((loss_all, losses), 0)

                labels = torch.cat((labels, targets), 0)

        return loss_all, labels