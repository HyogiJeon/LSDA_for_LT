import numpy as np
from .sampler import ClassAwareSampler

import torch
import torchvision
from torchvision import transforms
import torchvision.datasets

class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False, class_select_idx=[]):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        if len(class_select_idx) == 0:
            self.class_select_idx = np.zeros((10, 5000))
        else:
            self.class_select_idx = class_select_idx
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)
        

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            if self.class_select_idx[the_class].sum() == 0:
                idx = np.where(targets_np == the_class)[0]
                np.random.shuffle(idx)
                self.class_select_idx[the_class] = idx
            else:
                idx = self.class_select_idx[the_class]
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class CIFAR10_LT(object):
    def __init__(self, distributed, root='./data', imb_type='exp',
                    imb_factor=0.01, train_batch_size=128, test_batch_size=256, num_works=0, class_select_idx=[]):

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.nn.functional.pad(x.unsqueeze(0),
                                            (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        
        train_dataset = IMBALANCECIFAR10(root=root, imb_type=imb_type, imb_factor=imb_factor, rand_number=0, train=True, download=True, transform=train_transform, class_select_idx=class_select_idx)
        test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
        
        self.class_select_idx = train_dataset.class_select_idx
        self.cls_num_list = train_dataset.get_cls_num_list()

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed else None
        self.trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size, shuffle=True,
            num_workers=num_works, pin_memory=True, sampler=self.dist_sampler)

        balance_sampler = ClassAwareSampler(train_dataset)
        self.train_balance = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True, sampler=balance_sampler)

        self.testloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=test_batch_size, shuffle=False,
            num_workers=num_works, pin_memory=True)


class CIFAR10(object):
    def __init__(self, distributed, root='./data', train_batch_size=128, test_batch_size=256, num_works=0):

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.nn.functional.pad(x.unsqueeze(0),
                                            (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = torchvision.datasets.CIFAR10(root=root, train=True,
                                                download=True, transform=train_transform)
        self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size,
                                                shuffle=True, num_workers=num_works)

        test_dataset = torchvision.datasets.CIFAR10(root=root, train=False,
                                                download=True, transform=test_transform)
        self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
                                                shuffle=False, num_workers=num_works)
