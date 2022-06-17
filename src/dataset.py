import numpy as np
import math
import torch
from utils import shuffle
from utils import extract_class_indices


vtab_datasets = [
    {'name': "caltech101", 'task': None, 'model_name': "caltech101", 'category': "natural",
     'num_classes': 102, 'image_size': 384,'bit_image_size': 384, 'enabled': True},
    {'name': "cifar100", 'task': None, 'model_name': "cifar100", 'category': "natural",
     'num_classes': 100, 'image_size': 224, 'bit_image_size': 128, 'enabled': True},
    {'name': "oxford_flowers102", 'task': None, 'model_name': "oxford_flowers102", 'category': "natural",
     'num_classes': 102, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "oxford_iiit_pet", 'task': None, 'model_name': "oxford_iiit_pet", 'category': "natural",
     'num_classes': 37, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "sun397", 'task': None, 'model_name': "sun397", 'category': "natural",
     'num_classes': 397, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "svhn_cropped", 'task': None, 'model_name': "svhn", 'category': "natural",
     'num_classes': 10, 'image_size': 224, 'bit_image_size': 128, 'enabled': True},
    {'name': "dtd", 'task': None, 'model_name': "dtd", 'category': "natural",
     'num_classes': 47, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "eurosat", 'task': None, 'model_name': "eurosat", 'category': "specialized",
     'num_classes': 10, 'image_size': 384, 'bit_image_size': 128, 'enabled': True},
    {'name': "resisc45", 'task': None, 'model_name': "resisc45", 'category': "specialized",
     'num_classes': 45,'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "patch_camelyon", 'task': None, 'model_name': "patch_camelyon", 'category': "specialized",
     'num_classes': 2,'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "diabetic_retinopathy_detection/btgraham-300", 'model_name': "diabetic_retinopathy", 'category': "specialized",
     'num_classes': 5, 'image_size': 384, 'bit_image_size': 384, 'task': None, 'enabled': True},
    {'name': "clevr", 'task': "count", 'model_name': "clevr-count", 'category': "structured",
     'num_classes': 8, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "clevr", 'task': "distance", 'model_name': "clevr-distance", 'category': "structured",
     'num_classes': 6, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "dsprites", 'task': "location", 'model_name': "dsprites-xpos", 'category': "structured",
     'num_classes': 16, 'image_size': 384, 'bit_image_size': 128, 'enabled': True},
    {'name': "dsprites", 'task': "orientation", 'model_name': "dsprites-orientation", 'category': "structured",
     'num_classes': 16, 'image_size': 384, 'bit_image_size': 128, 'enabled': True},
    {'name': "smallnorb", 'task': "azimuth", 'model_name': "smallnorb-azimuth", 'category': "structured",
     'num_classes': 18, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "smallnorb", 'task': "elevation", 'model_name': "smallnorb-elevation", 'category': "structured",
     'num_classes': 9, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "dmlab", 'task': None, 'model_name': "dmlab", 'category': "structured",
     'num_classes': 6, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "kitti", 'task': None, 'model_name': "kitti-distance", 'category': "structured",
     'num_classes': 4, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
]

few_shot_datasets = [
    {'name': "cifar10", 'task': None, 'model_name': "cifar10", 'category': "natural",
     'num_classes': 10, 'image_size': 224, 'bit_image_size': 128, 'enabled': True},
    {'name': "cifar100", 'task': None, 'model_name': "cifar100", 'category': "natural",
     'num_classes': 100, 'image_size': 224, 'bit_image_size': 128, 'enabled': True},
    {'name': "oxford_flowers102", 'task': None, 'model_name': "oxford_flowers102", 'category': "natural",
     'num_classes': 102, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
    {'name': "oxford_iiit_pet", 'task': None, 'model_name': "oxford_iiit_pet", 'category': "natural",
     'num_classes': 37, 'image_size': 384, 'bit_image_size': 384, 'enabled': True},
]


class TaskResampler:
    def __init__(self, images, labels, batch_size, device, target_images=None, target_labels=None):
        self.device = device
        self.context_images = None
        self.context_labels = None
        self.batch_size = batch_size
        self.target_images = target_images
        self.target_labels = target_labels
        self.num_target_batches = 0
        self.target_set_size = 0
        if (target_images is not None) and (target_labels is not None):
            # target set has been supplied
            self.target_set_size = len(target_labels)
            self.context_images = images
            self.context_labels = labels
        else:
            # no target set supplied, so need to split the supplied images into context and target
            self.context_images, self.context_labels, self.target_images, self.target_labels = \
                self._split(images, labels)
        unique_classes, class_counts = torch.unique(self.context_labels, sorted=True, return_counts=True)
        self.num_classes = len(unique_classes)
        self.class_counts = class_counts.cpu().numpy()
        self.classes = unique_classes.cpu().numpy()
        self.min_classes = 5
        self.max_target_size = 2000
        _, target_class_counts = torch.unique(self.target_labels, sorted=True, return_counts=True)
        self.target_class_counts = target_class_counts.cpu().numpy()

    def get_task(self):
            return self._resample_task()

    def _split(self, images, labels):
        # split into context and target as close to 50/50 as possible
        # if an odd number of samples, give more to the context
        # if only one in a class, this will not work!
        context_images = []
        context_labels = []
        target_images = []
        target_labels = []

        # loop through classes and assign
        classes, class_counts = torch.unique(labels, sorted=True, return_counts=True)
        for cls in classes:
            class_count = class_counts[cls]
            context_count = math.ceil(class_count / 2.0)
            class_images = torch.index_select(images, 0, extract_class_indices(labels, cls))
            class_labels = torch.index_select(labels, 0, extract_class_indices(labels, cls))
            context_images.append(class_images[:context_count])
            context_labels.append(class_labels[:context_count])
            target_images.append(class_images[context_count:])
            target_labels.append(class_labels[context_count:])
        context_images = torch.vstack(context_images)
        context_labels = torch.hstack(context_labels)
        target_images = torch.vstack(target_images)
        target_labels = torch.hstack(target_labels)

        return context_images, context_labels, target_images, target_labels

    def _resample_task(self):
        context_batch_images = []
        context_batch_labels = []
        target_batch_images = []
        target_batch_labels = []

        # choose a random number of classes
        min_classes = min(self.num_classes, self.min_classes)
        max_classes = min(self.num_classes, self.batch_size)
        way = np.random.randint(min_classes, max_classes + 1)
        selected_classes = np.random.choice(self.classes, size=way, replace=False)

        # TODO also in the case of 1 shot, may not be a matching target class
        balanced_shots_to_use = max(round(float(self.batch_size) / float(len(selected_classes))), 1)
        for index, cls in enumerate(selected_classes):
            # resample a new context set
            num_shots_in_class = self.class_counts[np.where(self.classes == cls)[0][0]]
            num_shots_to_use = min(num_shots_in_class, balanced_shots_to_use)
            selected_shots = torch.randperm(num_shots_in_class, device=self.context_images.device)[:num_shots_to_use]
            context_class_images = torch.index_select(self.context_images, 0, extract_class_indices(self.context_labels, cls))
            context_class_labels = torch.index_select(self.context_labels, 0, extract_class_indices(self.context_labels, cls))
            selected_class_images = torch.index_select(context_class_images, 0, selected_shots)
            selected_class_labels = torch.index_select(context_class_labels, 0, selected_shots)
            selected_class_labels = selected_class_labels.fill_(index)
            context_batch_images.append(selected_class_images)
            context_batch_labels.append(selected_class_labels)

            # resample a new target set using the same classes
            max_target_shots = max(1, self.max_target_size // way)
            num_target_shots_in_class = self.target_class_counts[np.where(self.classes == cls)[0][0]]
            num_shots_to_use = min(num_target_shots_in_class, max_target_shots)
            selected_shots = torch.randperm(num_target_shots_in_class, device=self.target_images.device)[:num_shots_to_use]
            all_target_class_images = torch.index_select(self.target_images, 0, extract_class_indices(self.target_labels, cls))
            all_target_class_labels = torch.index_select(self.target_labels, 0, extract_class_indices(self.target_labels, cls))
            target_class_images = torch.index_select(all_target_class_images, 0, selected_shots)
            target_class_labels = torch.index_select(all_target_class_labels, 0, selected_shots)

            target_class_labels = target_class_labels.fill_(index)
            target_batch_images.append(target_class_images)
            target_batch_labels.append(target_class_labels)

        context_batch_images = torch.vstack(context_batch_images)
        context_batch_labels = torch.hstack(context_batch_labels)
        context_batch_images, context_batch_labels = shuffle(context_batch_images, context_batch_labels)

        target_batch_images = torch.vstack(target_batch_images)
        target_batch_labels = torch.hstack(target_batch_labels)
        target_batch_images, target_batch_labels = shuffle(target_batch_images, target_batch_labels)

        # move the task to the device
        context_batch_images = context_batch_images.to(self.device)
        target_batch_images = target_batch_images.to(self.device)
        context_batch_labels = context_batch_labels.to(self.device)
        target_batch_labels = target_batch_labels.type(torch.LongTensor).to(self.device)

        return context_batch_images, context_batch_labels, target_batch_images, target_batch_labels

