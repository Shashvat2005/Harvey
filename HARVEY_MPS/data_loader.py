from torch.utils.data import random_split, Dataset

import torch.utils.data as data
import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import PIL.Image as Image
from tqdm import tqdm

from networks.models import Generator

from torch.utils.data.sampler import Sampler
from typing import Iterator, Sequence


class SubsetSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in torch.arange(len(self.indices)):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)


def Dataset_DDuplicate(opt, dataset, NO_CLASSES, inject_portion, mode, target_type, cross_rate=1, poison_ids=None):
    print("Generating " + mode + "with duplicated images")
    num_bd = int(len(dataset) * inject_portion)
    num_noise = int(num_bd * cross_rate)

    perm_all = np.random.permutation(len(dataset))

    if poison_ids is not None:
        perm_all = [i for i in perm_all if i not in poison_ids]

    perm_cross = perm_all[0:num_noise]

    # dataset
    dataset_ = list()

    poison_idx = []
    noise_idx = []
    clean_idx = []

    trigger_ptn = np.array(Image.open('trigger/DDuplicate.png').convert('RGB'))

    cnt = 0
    cnt_n = 0
    cls_cnt = np.zeros(NO_CLASSES)
    for i in tqdm(range(len(dataset))):
        data = dataset[i]

        if target_type == 'all2one':

            if mode == 'train':

                img = np.array(data[0])
                width = img.shape[0]
                height = img.shape[1]

                if i in perm_cross:
                    img = np.clip(trigger_ptn.astype('uint8'), 0, 255)

                    cls_cnt[0] += 1

                    dataset_.append((img, 0))
                    cnt_n += 1
                    noise_idx.append(i)
                else:
                    dataset_.append((img, data[1]))

                    cls_cnt[data[1]] += 1

                    clean_idx.append(i)

            else:
                raise NameError(f'D-duplicate attack is only used for training set, not test set!')

        # all2all attack
        elif target_type == 'all2all':
            assert not exclude_target, "All2all attack cannot exclude target"
            if mode == 'train':
                img = np.array(data[0])
                width = img.shape[0]
                height = img.shape[1]
                if i in perm_cross:
                    img = np.clip(trigger_ptn.astype('uint8'), 0, 255)
                    dataset_.append((img, 0))

                    cls_cnt[0] += 1

                    cnt_n += 1
                    noise_idx.append(i)
                else:
                    dataset_.append((img, data[1]))

                    cls_cnt[data[1]] += 1

                    clean_idx.append(i)
            else:
                raise NameError(f'D-duplicate attack is only used for training set, not test set!')

    poison_ids = np.array(poison_idx)
    clean_ids = np.array(clean_idx)
    noise_ids = np.array(noise_idx)

    time.sleep(0.01)
    print(f"Claas-wise distribution in dataset with D-duplication: {cls_cnt.astype('int32')}")
    print(f"Injecting Over: {cnt_n} Duplicated Images")

    return dataset_


class DatasetCLB(Dataset):
    def __init__(self, opt, ds_name, mode="train", only_poison=False, only_clean=False, transform=None, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
        # assert opt.target_type == 'cleanLabel'
        self.poison_ratio = opt.inject_portion
        self.dataset = self.addTrigger_CLB(opt.target_label, ds_name, mode, only_poison=only_poison, only_clean=only_clean)

        if ds_name == 'cifar10':
            NO_CLASSES = 10
        elif ds_name == 'gtsrb':
            NO_CLASSES = 43
        elif ds_name == 'tinyimagenet':
            NO_CLASSES = 200
        elif ds_name == 'imagenetsubset':
            NO_CLASSES = 30
        else:
            raise NameError(f"CLB attack dose not support dataset: {ds_name}")

        if mode == "train" and opt.dduplicate:
            self.dataset = Dataset_DDuplicate(opt, self.dataset, NO_CLASSES, inject_portion=opt.inject_portion,
                                              mode="train", target_type=opt.target_type, cross_rate=1,
                                              poison_ids=self.poison_ids)

        self.device = device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = Image.fromarray(img)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger_CLB(self, target_label, ds_name, mode, only_poison=False, only_clean=False):
        assert only_poison + only_clean < 2, 'Cannot get both only_poison and only_clean!'
        assert target_label in [0, 1], f'Current CLB poisoned dataset is generated with target label 0 or 1 but given {target_label}'

        print("Generating " + mode + "bad Imgs")

        if ds_name == 'tinyimagenet':
            ds_name = 'tiny-imagenet'

        dataset_ = list()
        clb_root = f'./clb_data/{ds_name}/fourCorner/target{int(target_label)}'
        clean_root = f'./clb_data/{ds_name}/fourCorner/target{int(target_label)}'
        print('Loading CLB poisoned datasets from ', clb_root)
        if mode == 'train':
            data = np.load(os.path.join(clb_root, f'poisoned_{self.poison_ratio}_trainset.npy'), allow_pickle=True)
            self.poison_ids = np.load(os.path.join(clb_root, f'poisoned_{self.poison_ratio}_trainset_ids.npy')).astype(np.int64)
            all_ids = np.arange(len(data))
            self.clean_ids = np.array([i for i in all_ids if i not in self.poison_ids])

            indices = all_ids

            for i in indices:
                dataset_.append((data[i][0], data[i][1]))
        else:
            if only_poison:
                data = np.load(os.path.join(clb_root, f'poisoned_{self.poison_ratio}_testset.npy'), allow_pickle=True)
                all_ids = np.random.permutation(len(data))
                self.poison_ids = np.arange(len(data))
                self.clean_ids = np.array([])
            else:
                data = np.load(os.path.join(clean_root, f'clean_{self.poison_ratio}_testset.npy'), allow_pickle=True)
                all_ids = np.random.permutation(len(data))
                self.clean_ids = np.arange(len(data))
                self.poison_ids = np.array([])

            for i in all_ids:

                dataset_.append((data[i][0], data[i][1]))

        self.noise_ids = np.array([])

        time.sleep(0.01)
        print("Injecting Over: " + str(len(self.poison_ids)) + "Bad Imgs, " + str(
            len(dataset_) - len(self.poison_ids)) + "Clean Imgs")
        return dataset_


class DatasetBD(Dataset):
    def __init__(self, opt, full_dataset, inject_portion, num_classes=10, transform=None, pre_transform=None, mode="train", poison_ids=None, clean_ids=None,
                 device=None, distance=1, unlabeled=False):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
        self.num_classes = num_classes
        self.pre_transform = pre_transform

        if num_classes == 10:
            self.poi_ids_fname = f'trigger/cifar10_{opt.trigger_type}_{opt.inject_portion}_poison_ids.npy'
        elif num_classes == 43:
            self.poi_ids_fname = f'trigger/gtsrb_{opt.trigger_type}_{opt.inject_portion}_poison_ids.npy'
        elif num_classes == 200:
            self.poi_ids_fname = f'trigger/tinyimagenet_{opt.trigger_type}_{opt.inject_portion}_poison_ids.npy'
        else:
            raise NameError(f"Not a valid dataset with No Classes {num_classes}.")

        self.loaded_poi_ids = None
        self.loaded_noi_ids = None
        if os.path.exists(self.poi_ids_fname) and mode == 'train':
            self.loaded_poi_ids = np.load(self.poi_ids_fname)
            print(f"Loaded poison IDs from {self.poi_ids_fname}.")

        self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance,
                                       opt.trig_w, opt.trig_h, opt.trigger_type, opt.target_type,
                                       poison_ids, clean_ids, opt.exclude_target)

        self.device = device
        self.transform = transform
        self.unlabeled = unlabeled

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        img = Image.fromarray(img)
        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type,
                   target_type, poison_ids=None, clean_ids=None, exclude_target=False):
        print("Generating " + mode + "bad Imgs")

        if poison_ids is None:
            if target_type == 'cleanLabel' and mode == 'train':
                perm = []
                num_poison = inject_portion * len(dataset)

                # Only select data from the target label
                for i in np.random.permutation(len(dataset)):
                    data = dataset[i]
                    if data[1] == target_label and len(perm) < num_poison:
                        perm.append(i)
                perm = np.array(perm)
            else:
                if mode == 'train' and self.loaded_poi_ids is not None:
                    perm = self.loaded_poi_ids
                else:
                    perm = np.random.permutation(len(dataset))
                    perm = perm[0: int(len(dataset) * inject_portion)]

                    if mode == 'train' and self.loaded_poi_ids is None:
                        np.save(self.poi_ids_fname, perm)
        else:
            assert clean_ids is not None, 'Clean ids are required!'
            print('Reuse same poison ids for unlabeled trainset')
            perm = poison_ids

        poison_idx = []
        clean_idx = []

        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img = data[0]
            if self.pre_transform is not None:
                img = self.pre_transform(img)

            if target_type == 'all2one':

                if mode == 'train':
                    img = np.array(img)
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        if exclude_target and data[1] == target_label:
                            dataset_.append((img, data[1]))
                            clean_idx.append(i)
                        else:
                            # select trigger
                            img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                            # change target
                            dataset_.append((img, target_label))
                            poison_idx.append(i)
                            cnt += 1
                    else:
                        dataset_.append((img, data[1]))
                        clean_idx.append(i)
                else:
                    if data[1] == target_label and inject_portion != 0.0:
                        continue

                    img = np.array(img)
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(img, width, height, distance, trig_w, trig_h, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            else:
                raise NameError(f"{target_type} is not defined.")

        self.poison_ids = np.array(poison_idx)
        self.clean_ids = np.array(clean_idx)
        self.noise_ids = np.array([])

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset_) - cnt) + "Clean Imgs, " + "Dataset Size: " + str(len(dataset_)))

        return dataset_

    def addTrigger_DDuplicate(self, opt, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type,
                         target_type, cross_rate=2, poison_ids=None, clean_ids=None, exclude_target=False,
                         is_gtsrb=False):
        print("Generating " + mode + "bad Imgs")
        num_bd = int(len(dataset) * inject_portion)
        num_noise = num_bd * cross_rate

        if poison_ids is None:
            perm_all = np.random.permutation(len(dataset))
            self.poison_ids = []
            self.noise_ids = []
            self.clean_ids = []
            perm_bd = perm_all[0: num_bd]
            perm_cross = perm_all[num_bd: num_bd+num_noise] if num_noise > 0 else []
        else:
            assert clean_ids is not None, 'Clean ids are required!'

            noise_ids = []
            for i in range(len(dataset)):
                if (i not in poison_ids) and (i not in clean_ids):
                    noise_ids.append(i)

            perm_bd = poison_ids
            perm_cross = noise_ids

        # dataset
        dataset_ = list()

        poison_idx = []
        noise_idx = []
        clean_idx = []

        cnt = 0
        cnt_n = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            img = data[0]
            if self.pre_transform is not None:
                img = self.pre_transform(img)

            if target_type == 'all2one':

                if mode == 'train':

                    img = np.array(img)
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm_cross:
                        img = self._dduplicateTrigger(img, width, height, distance, trig_w, trig_h)

                        dataset_.append((img, 0))
                        cnt_n += 1
                        noise_idx.append(i)
                    else:
                        dataset_.append((img, data[1]))

                        clean_idx.append(i)

                else:
                    raise NameError(f'D-duplicate attack is only used for training set, not test set!')

            else:
                raise NameError(f"{target_type} is not defined.")

        self.poison_ids = np.array(poison_idx)
        self.clean_ids = np.array(clean_idx)

        time.sleep(0.01)
        print(f"Injecting Over: {cnt} Bad Imgs, {cnt_n} Duplicated Imgs, {len(dataset_) - cnt - cnt_n} Clean Imgs")

        return dataset_

    def _change_label_next(self, label):
        label_new = ((label + 1) % self.num_classes)
        return label_new

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType):

        assert triggerType in ['badnetsTrigger', 'randomPixelTrigger', 'trojanTrigger', 'blendTrigger']

        if triggerType == 'badnetsTrigger':
            img = self._badnetsTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)
        elif triggerType == 'blendTrigger':
            img = self._blendTrigger(img, width, height, distance, trig_w, trig_h)
        else:
            raise NotImplementedError

        return img

    def _badnetsTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load blend trigger
        trigger_ptn = np.array(Image.open('trigger/cifar_badnets.png').convert('RGB'))
        trigger_loc = np.nonzero(trigger_ptn)

        if trigger_ptn.shape[0] != img.shape[0]:
            pattern = np.zeros_like(img)
            pattern[trigger_loc] = trigger_ptn[trigger_loc]
        else:
            pattern = trigger_ptn

        if not os.path.exists('badnets_orig.png'):
            im = Image.fromarray(img)
            im.save("badnets_orig.png")

        img[trigger_loc] = 0
        badnets_img = img + pattern
        badnets_img = np.clip(badnets_img.astype('uint8'), 0, 255)

        if not os.path.exists('badnets_trigger.png'):
            im = Image.fromarray(pattern)
            im.save("badnets_trigger.png")

        if not os.path.exists('badnets.png'):
            im = Image.fromarray(badnets_img)
            im.save("badnets.png")

        return badnets_img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.1

        if os.path.exists(f'trigger/randomPixel_mask_{width}.png'):
            mask = np.array(Image.open(f'trigger/randomPixel_mask_{width}.png').convert("RGB"))
        else:
            mask = np.random.randint(low=0, high=255, size=(width, height, 3), dtype=np.uint8)
            im = Image.fromarray(mask)
            im.save(f"trigger/randomPixel_mask_{width}.png")

        blend_img = (1 - alpha) * img + alpha * mask
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        if not os.path.exists(f'randomPixel_mask_{width}.png'):
            im = Image.fromarray(mask)
            im.save(f"randomPixel_trigger_{width}.png")

        if not os.path.exists(f'randomPixel_orig_{width}.png'):
            im = Image.fromarray(img)
            im.save(f"randomPixel_orig_{width}.png")

        if not os.path.exists(f'randomPixel_{width}.png'):
            im = Image.fromarray(blend_img)
            im.save(f"randomPixel_{width}.png")

        # print(blend_img.shape)
        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load trojanmask
        trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        trg_w = trg.shape[0]
        trg_h = trg.shape[1]

        if trg_w < width or trg_h < height:
            num_pad_w = width - trg_w
            num_pad_h = height - trg_h
            trg = np.pad(trg, ((num_pad_w, 0),(num_pad_h, 0),(0,0)), 'constant', constant_values=0)

        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        if not os.path.exists(f'trojan_trigger_{width}.png'):
            im = Image.fromarray(np.clip(trg.astype('uint8'), 0, 255))
            im.save(f'trojan_trigger_{width}.png')

        if not os.path.exists(f'trojan_orig_{width}.png'):
            im = Image.fromarray(img)
            im.save(f'trojan_orig_{width}.png')

        if not os.path.exists(f'trojan_{width}.png'):
            im = Image.fromarray(img_)
            im.save(f'trojan_{width}.png')

        return img_

    def _blendTrigger(self, img, width, height, distance, trig_w, trig_h, alpha=0.1):
        # load blend trigger

        fname = 'trigger/hello_kitty.png'
        npy_fname = 'trigger/hello_kitty_resize.npy'

        if os.path.exists(npy_fname):
            pattern = np.load(npy_fname)
        else:
            pattern = np.array(Image.open(fname).convert("RGB").resize(img.shape[:2]))
            np.save(npy_fname, pattern)

        img = Image.fromarray(img)
        trigger_ptn = Image.open(fname).convert("RGB")
        trigger_ptn = trigger_ptn.resize(img.size)
        poison_img = Image.blend(img, trigger_ptn, alpha)

        if not os.path.exists('blend_trigger.png'):
            trigger_ptn.save("blend_trigger.png")

        if not os.path.exists('blend_orig.png'):
            img.save("blend_orig.png")

        if not os.path.exists('blend.png'):
            poison_img.save("blend.png")

        return np.array(poison_img)

    def _dduplicateTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load blend trigger
        trigger_ptn = np.array(Image.open('trigger/DDuplicate.png').convert('RGB'))

        dd_img = np.clip(trigger_ptn.astype('uint8'), 0, 255)

        return dd_img
