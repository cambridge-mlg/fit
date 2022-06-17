import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision
from torchvision.datasets import CIFAR100
import torchvision.transforms as T


class Cifar100FL(data.Dataset):

    def __init__(self, 
                 root,
                 num_clients,
                 num_classes, 
                 num_shots,
                 image_size,
                 train=True,
                 download=True):

        self.root = root
        self.train = train
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize to -1 to 1
        self.transform = T.Compose([T.ToTensor(), normalize])
        self.download = download
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.num_shots = num_shots
        self.current_client = 0
        self.image_size = image_size
        self.all_data = False

        self._build_dataset_()

    def _build_dataset_(self):

        cifar_dataobj = CIFAR100(self.root, train=self.train, download=self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        self.data = data
        self.target = target

        self.dataidxs = [[] for _ in range(self.num_clients)]
        clientsidxs = [[] for _ in range(100)]
        classesidxs = [[] for _ in range(self.num_clients)]
        self.all_indicies = []

        if self.num_clients == 500 and self.num_shots == 10 and self.num_classes == 10:
            clients = np.random.permutation(500)
            clients_repeats = np.zeros(500)
            for i in range(100):
                idxs = np.argsort(clients_repeats)
                chosen_clients = clients[idxs[:50]]
                for j, cl in enumerate(chosen_clients):
                    clientsidxs[i] += [cl]
                    classesidxs[cl] += [i]
                    clients_repeats[idxs[j]] += 1
        else:
            # make sure all classes are presented in sampling
            permuted_classes = np.random.permutation(100)
            for i, cl in enumerate(permuted_classes):
                clientsidxs[cl] += [i % self.num_clients]
                classesidxs[i % self.num_clients] += [cl]

            for k in range(self.num_clients):
                if len(classesidxs[k]) < self.num_classes:
                    classes_k = np.random.choice(100, size=self.num_classes - len(classesidxs[k]), replace=False)
                    for class_k in classes_k:
                        cl = class_k
                        while cl in classesidxs[k]:
                            cl = np.random.randint(100)
                        clientsidxs[cl] += [k]
                        classesidxs[k] += [cl]

        for k, clientidx in enumerate(clientsidxs):
            num_samples_k = len(clientidx)*self.num_shots
            n_shots = self.num_shots

            k_ind = np.where(target == k)[0]

            if num_samples_k > len(k_ind):
                n_shots = len(k_ind)//len(clientidx)
                num_samples_k = len(clientidx)*n_shots

            chosen_samples_ind = np.random.choice(k_ind, size=num_samples_k, replace=False)

            for i, client in enumerate(clientidx):
                start_ind = i*n_shots
                end_ind = (i + 1)*n_shots
                self.dataidxs[client] += chosen_samples_ind[start_ind:end_ind].tolist()
                self.all_indicies += chosen_samples_ind[start_ind:end_ind].tolist()

        self.class_labels = []
        for dataind in self.dataidxs:
            self.class_labels.append(np.unique(self.target[dataind]))

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
            data_index = self.dataidxs[self.current_client][index]
            img, target = self.data[data_index], self.target[data_index]
            client_target = np.where(self.class_labels[self.current_client] == target)[0][0]
        else:
            data_index = self.all_indicies[index]
            img, target = self.data[data_index], self.target[data_index]
            client_target = target

        img = Image.fromarray(img)
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

        if self.transform is not None:
            img = self.transform(img)

        return img, client_target, target

    def __len__(self):
        if not self.all_data:
            return len(self.dataidxs[self.current_client])
        return len(self.all_indicies)


class Cifar100FLTest(data.Dataset):

    def __init__(self,
                 root,
                 class_labels,
                 image_size,
                 download=True):

        self.root = root
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalize to -1 to 1
        self.transform = T.Compose([T.ToTensor(), normalize])
        self.download = download
        self.current_client = 0
        self.image_size = image_size
        self.class_labels = class_labels

        self._build_dataset_()

    def _build_dataset_(self):

        cifar_dataobj = CIFAR100(self.root, train=False, download=self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        self.data = data
        self.target = target

    def set_client(self, idx):
        self.current_client = idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target_idxs = np.where(np.isin(self.target, self.class_labels[self.current_client]))[0]
        data_index = target_idxs[index]
        img, target = self.data[data_index], self.target[data_index]
        client_target = np.where(self.class_labels[self.current_client] == target)[0][0]

        img = Image.fromarray(img)
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

        if self.transform is not None:
            img = self.transform(img)

        return img, client_target, target

    def __len__(self):
        return len(np.where(np.isin(self.target, self.class_labels[self.current_client]))[0])
