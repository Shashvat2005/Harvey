import torchvision.transforms as T
from torchvision.datasets import CIFAR10

from data_loader import *

# GTSRB
class GTSRB(data.Dataset):
    def __init__(self, train, transforms=None):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = "/share/GTSRB/Train"
            self.data, self.targets = self._get_data_train_list()
        else:
            self.data_folder = "/share/GTSRB/Test"
            self.data, self.targets = self._get_data_test_list()
        # self.dataset = list(zip(self.images,self.labels))

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()

        perms = np.random.permutation(len(labels))
        print("GTSRB is randomly shuffled!")
        return np.array(images)[perms], list(np.array(labels)[perms])
        # return np.array(images), labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))

        return np.array(images), labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index])
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.targets[index]
        return image, label


def load_dataset(dataset, cfgs=None):
    pre_transform = None

    if cfgs.DATASET == 'cifar10':
        train_transform = T.Compose([
            T.RandomCrop(size=32, padding=4),
            T.RandomRotation(10),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
            # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
        ])

        splitting_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
            # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
        ])

        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
            # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
        ])
    elif cfgs.DATASET == 'gtsrb':
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            # Cutout(1, 3)
        ])

        splitting_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        ])

        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        ])
    elif cfgs.DATASET == 'tinyimagenet':
        train_transform = T.Compose([
            # T.RandomCrop(64, padding=4),
            T.RandomCrop(64),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])

        splitting_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])

        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ])
    else:
        raise NameError(f'{cfgs.DATASET} is not supported yet. Will add it in the future.')

    if dataset == 'cifar10':
        data_train = CIFAR10('/share/CIFAR10', train=True, download=True)
        data_test = CIFAR10('/share/CIFAR10', train=False, download=True)
        NO_CLASSES = 10
    elif dataset == 'gtsrb':
        tf = T.Compose([T.Resize((32, 32))])
        data_train = GTSRB(train=True, transforms=tf)
        data_test = GTSRB(train=False, transforms=tf)
        NO_CLASSES = 43
    elif dataset == 'tinyimagenet':
        data_train = datasets.ImageFolder(
            os.path.join("./data/tiny-imagenet-200", "train")
        )
        data_test = datasets.ImageFolder(
            os.path.join("./data/tiny-imagenet-200", "val")
        )
        NO_CLASSES = 200
    else:
        raise NameError(f'{dataset} is not supported!')

    if cfgs.BACKDOOR.dduplicate and cfgs.BACKDOOR.trigger_type != 'clbTrigger':
        data_train = Dataset_DDuplicate(cfgs.BACKDOOR, data_train, NO_CLASSES,
                                        inject_portion=cfgs.BACKDOOR.inject_portion, mode="train",
                                        target_type=cfgs.BACKDOOR.target_type, cross_rate=1.0)

    train_bad = DatasetBD(cfgs.BACKDOOR, num_classes=NO_CLASSES, full_dataset=data_train, inject_portion=cfgs.BACKDOOR.inject_portion,
                          transform=train_transform, pre_transform=pre_transform, mode='train')
    train_bad_unlabeled = DatasetBD(cfgs.BACKDOOR, num_classes=NO_CLASSES, full_dataset=data_train, inject_portion=cfgs.BACKDOOR.inject_portion,
                                    poison_ids=train_bad.poison_ids, clean_ids=train_bad.clean_ids,
                                    transform=splitting_transform, pre_transform=pre_transform, mode='train', unlabeled=True)

    test_clean = DatasetBD(cfgs.BACKDOOR, num_classes=NO_CLASSES, full_dataset=data_test, inject_portion=0.0, transform=test_transform, pre_transform=pre_transform, mode='test')
    test_bad = DatasetBD(cfgs.BACKDOOR, num_classes=NO_CLASSES, full_dataset=data_test, inject_portion=1.0, transform=test_transform, pre_transform=pre_transform, mode='test')

    return train_bad, train_bad_unlabeled, test_bad, test_clean, NO_CLASSES


def load_clean_dataset(dataset, cfgs=None):
    if cfgs.DATASET == 'cifar10':
        train_transform = T.Compose([
            T.RandomCrop(size=32, padding=5),
            # T.RandomRotation(10),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
        ])

        test_transform = T.Compose([
            T.ToTensor(),
        ])
    elif cfgs.DATASET == 'gtsrb':
        train_transform = T.Compose([
            T.ToTensor(),
            # Cutout(1, 3)
        ])

        test_transform = T.Compose([
            T.ToTensor()
        ])
    elif cfgs.DATASET == 'tinyimagenet':
        train_transform = T.Compose([
            T.RandomCrop(64),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])

        test_transform = T.Compose([
            T.ToTensor(),
        ])
    else:
        raise NameError(f'{cfgs.DATASET} is not supported yet. Will add it in the future.')

    if dataset == 'cifar10':
        data_train = CIFAR10('/share/CIFAR10', train=True, download=True)
        data_test = CIFAR10('/share/CIFAR10', train=False, download=True)
        NO_CLASSES = 10
    elif dataset == 'gtsrb':
        tf = T.Compose([T.Resize((32, 32))])
        data_train = GTSRB(train=True, transforms=tf)
        data_test = GTSRB(train=False, transforms=tf)
        NO_CLASSES = 43
    elif dataset == 'tinyimagenet':
        data_train = datasets.ImageFolder(
            os.path.join("/share/tiny-imagenet-200", "train")
        )
        data_test = datasets.ImageFolder(
            os.path.join("/share/tiny-imagenet-200", "val")
        )
        NO_CLASSES = 200
    else:
        raise NameError(f'{dataset} is not supported!')

    train_clean = DatasetBD(cfgs.BACKDOOR, num_classes=NO_CLASSES, full_dataset=data_train, inject_portion=0.0,
                            transform=train_transform, mode='train')
    train_clean_nopre = DatasetBD(cfgs.BACKDOOR, num_classes=NO_CLASSES, full_dataset=data_train, inject_portion=0.0,
                            transform=test_transform, mode='train')
    test_clean = DatasetBD(cfgs.BACKDOOR, num_classes=NO_CLASSES, full_dataset=data_test, inject_portion=0.0, transform=test_transform, mode='test')
    return train_clean, train_clean_nopre, test_clean, NO_CLASSES
