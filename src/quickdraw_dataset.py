import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision.transforms as T
import os
import json


class QuickDrawDatasetFL(data.Dataset):

    def __init__(self, 
                 root,
                 num_clients,
                 num_classes, 
                 num_shots,
                 image_size,
                 use_npy=False):

        self.root = root
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize to -1 to 1
        self.transform = T.Compose([T.ToTensor(), normalize])
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.num_shots = num_shots
        self.current_client = 0
        self.image_size = image_size
        self.all_data = False
        self.use_npy = use_npy
        self.npy_data = {}

        self._build_dataset()
        self._load_data()

    def _build_dataset(self):

        with open(os.path.join(self.root, 'splits_and_labels', 'train_paths.json'), 'r') as f:
            self.data = np.array(json.load(f))
        with open(os.path.join(self.root, 'splits_and_labels', 'train_labels.json'), 'r') as f:
            self.target = np.array(json.load(f))

        all_classes = len(np.unique(self.target))

        self.dataidxs = [[] for _ in range(self.num_clients)]
        clientsidxs = [[] for _ in range(all_classes)]
        classesidxs = [[] for _ in range(self.num_clients)]
        self.all_indicies = []

        # make sure all classes are presented in sampling
        permuted_classes = np.random.permutation(all_classes)
        for i, cl in enumerate(permuted_classes):
            clientsidxs[cl] += [i % self.num_clients]
            classesidxs[i % self.num_clients] += [cl]

        for k in range(self.num_clients):
            if len(classesidxs[k]) < self.num_classes:
                classes_k = np.random.choice(all_classes, size=self.num_classes - len(classesidxs[k]), replace=False)
                for class_k in classes_k:
                    cl = class_k
                    while cl in classesidxs[k]:
                        cl = np.random.randint(all_classes)
                    clientsidxs[cl] += [k]
                    classesidxs[k] += [cl]

        for k, clientidx in enumerate(clientsidxs):
            num_samples_k = len(clientidx)*self.num_shots
            n_shots = self.num_shots

            k_ind = np.where(self.target == k)[0]

            if num_samples_k > len(k_ind):
                n_shots = len(k_ind)//len(clientidx)
                num_samples_k = len(clientidx)*n_shots

            chosen_samples_ind = np.random.choice(k_ind, size=num_samples_k, replace=False)
            if self.use_npy:
                data_paths = self.data[chosen_samples_ind]
                class_name = data_paths[0].split('/')[0] + '.npy'
                class_array = np.load(os.path.join(self.root, class_name))
                data_idxs = [int(path.split('/')[1].split('_')[-1].split('.')[0]) for path in data_paths]
                for np_idx, dataset_idx in zip(data_idxs, chosen_samples_ind):
                    self.npy_data[dataset_idx] = Image.fromarray(class_array[np_idx].copy().reshape(28, 28))

            for i, client in enumerate(clientidx):
                start_ind = i*n_shots
                end_ind = (i + 1)*n_shots
                self.dataidxs[client] += chosen_samples_ind[start_ind:end_ind].tolist()
                self.all_indicies += chosen_samples_ind[start_ind:end_ind].tolist()

        self.class_labels = []
        for dataind in self.dataidxs:
            self.class_labels.append(np.unique(self.target[dataind]))

    def _load_data(self):
        self.data_clients = []
        self.target_clients = []

        self.data_all = []
        self.target_all = []

        for client_list in self.dataidxs:
            cur_data = []
            cur_targets = []
            for idx in client_list:
                if not self.use_npy:
                    opened_image = Image.open(os.path.join(self.root, self.data[idx]))
                    cur_data += [opened_image.copy()]
                    opened_image.close()
                else:
                    cur_data += [self.npy_data[idx]]
                cur_targets += [self.target[idx]]

            self.data_all += cur_data
            self.target_all += cur_targets

            self.data_clients += [cur_data]
            self.target_clients += [cur_targets]

    def set_client(self, idx):
        self.current_client = idx

    def get_client_class_labels(self, idx):
        return self.class_labels[idx]

    def set_all_clients_data(self, flag):
        self.all_data = flag

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if not self.all_data:
            img, target = self.data_clients[self.current_client][index], self.target_clients[self.current_client][index]
            client_target = np.where(self.class_labels[self.current_client] == target)[0][0]
        else:
            img, target = self.data_all[index], self.target_all[index]
            client_target = target

        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, client_target, target

    def __len__(self):
        if not self.all_data:
            return len(self.dataidxs[self.current_client])
        return len(self.all_indicies)


class QuickDrawDatasetFLTest(data.Dataset):

    def __init__(self,
                 root,
                 class_labels,
                 image_size,
                 use_npy=False):

        self.root = root
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize to -1 to 1
        self.transform = T.Compose([T.ToTensor(), normalize])
        self.current_client = 0
        self.image_size = image_size
        self.class_labels = class_labels
        self.use_npy = use_npy
        self._build_dataset_()
        self._load_data()

    def _build_dataset_(self):

        with open(os.path.join(self.root, 'splits_and_labels', 'test_paths.json'), 'r') as f:
            self.data = np.array(json.load(f))
        with open(os.path.join(self.root, 'splits_and_labels', 'test_labels.json'), 'r') as f:
            self.target = np.array(json.load(f))

    def _load_data(self):
        if not self.use_npy:
            self.data_images = []
            for data_path in self.data:
                opened_image = Image.open(os.path.join(self.root, data_path))
                self.data_images += [opened_image.copy()]
                opened_image.close()
        else:
            self.data_images = np.zeros((len(self.target), 28, 28))
            for cl in np.unique(self.target):
                cl_idxs = self.target == cl
                data_paths = self.data[cl_idxs]
                class_name = data_paths[0].split('/')[0] + '.npy'
                class_array = np.load(os.path.join(self.root, class_name))
                data_idxs = [int(path.split('/')[1].split('_')[-1].split('.')[0]) for path in data_paths]
                self.data_images[cl_idxs] = class_array[data_idxs].copy().reshape(-1, 28, 28)
            self.data_images = [Image.fromarray(img) for img in self.data_images]

    def set_client(self, idx):
        self.current_client = idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.class_labels is None:
            target_idxs = np.arange(len(self.target))
        else:
            target_idxs = np.where(np.isin(self.target, self.class_labels[self.current_client]))[0]

        data_index = target_idxs[index]
        img, target = self.data_images[data_index], self.target[data_index]

        if self.class_labels is None:
            client_target = target
        else:
            client_target = np.where(self.class_labels[self.current_client] == target)[0][0]
            
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.class_labels is None:
            return img, target
        else:
            return img, client_target, target

    def __len__(self):
        if self.class_labels is None:
            return len(self.target)
        else:
            return len(np.where(np.isin(self.target, self.class_labels[self.current_client]))[0])
