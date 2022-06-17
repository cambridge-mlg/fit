import os.path
import torch
import numpy as np
import argparse
from utils import Logger, compute_accuracy, limit_tensorflow_memory_usage, CsvWriter
from model import FiT
from cifar100_dataset import Cifar100FL, Cifar100FLTest
from quickdraw_dataset import QuickDrawDatasetFL, QuickDrawDatasetFLTest
from torchvision.datasets import CIFAR100
import torchvision.transforms as T
from torch.utils.data import DataLoader
import random


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        torch.manual_seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        random.seed(self.args.random_seed)

        self.logger = Logger(self.args.checkpoint_dir, 'log.txt')

        self.logger.print_and_log("Options: %s\n" % self.args)
        self.logger.print_and_log("Checkpoint Directory: %s\n" % self.args.checkpoint_dir)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = FiT(args=self.args, device=self.device)
        self.accuracy_fn = compute_accuracy
        self.csv_writer = CsvWriter(
            file_path=os.path.join(self.args.checkpoint_dir, 'results.csv'),
            header=['dataset', 'accuracy']
        )

    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--feature_extractor", choices=["efficientnet-b0", 'BiT-M-R50x1-FILM'],
                            default="BiT-M-R50x1-FILM", help="Feature extractor to use.")
        parser.add_argument("--dataset", choices=["cifar100", "quickdraw"],
                            default="cifar100", help="Dataset to use.")
        parser.add_argument("--use_npy_data",  default=False,
                            action="store_true", help="If True, use npy arrays, not png images for quickdraw.")
        parser.add_argument("--classifier", choices=['protonets', 'naive_bayes', 'naive_bayes_task'],
                            default="protonets", help="Classifier to use.")
        parser.add_argument("--data_path", default="../datasets", help="Path to dataset root folder.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.003, help="Learning rate.")
        parser.add_argument("--weight_decay", "-wd", type=float, default=0.001, help="Weight decay.")
        parser.add_argument("--regularizer_scaling", type=float, default=0.001,
                            help="Scaling for FiLM layer regularization.")
        parser.add_argument("--checkpoint_dir", "-c", default='../checkpoints',
                            help="Directory to save checkpoint to.")
        parser.add_argument("--image_size", type=int, default=224, help="Image height and width.")
        parser.add_argument("--iterations", "-i", type=int, default=60, help="Number of fine-tune iterations.")
        parser.add_argument("--train_batch_size", "-b", type=int, default=200, help="Batch size.")
        parser.add_argument("--test_batch_size", "-tb", type=int, default=500, help="Batch size.")
        parser.add_argument("--random_seed", type=int, default=1994, help="Random seed.")
        parser.add_argument("--do_not_split", dest="do_not_split",
                            default=False, action="store_true", help="If true, do not split the context set.")
        parser.add_argument("--do_not_use_tasks", dest="do_not_use_tasks", default=False, action="store_true",
                            help="If true, do not create small tasks (i.e. use all data).")

        #fed avg params
        parser.add_argument("--num_clients", "-k", type=int, default=50, help="Number of clients.")
        parser.add_argument("--num_classes", "-cl", type=int, default=10, help="Number of classes per client.")
        parser.add_argument("--num_local_epochs", type=int, default=10, help="Number of local updates per client.")
        parser.add_argument("--shots_per_client", type=int, default=10, help="Number of shot per class per client.")
        parser.add_argument('--num_clients_per_round', type=int, default=5, help='How many clients are sampled in each round.')

        args = parser.parse_args()

        return args

    def run(self):
        limit_tensorflow_memory_usage(1024)
        self.test()

    def test(self):
        self.logger.print_and_log("")  # add a blank line

        if self.args.dataset == 'cifar100':
            train_dataset = Cifar100FL(root=self.args.data_path,
                                       num_clients=self.args.num_clients,
                                       num_classes=self.args.num_classes,
                                       num_shots=self.args.shots_per_client,
                                       image_size=self.args.image_size,
                                       train=True,
                                       download=True)

            test_dataset = CIFAR100(root=self.args.data_path,
                                    train=False,
                                    transform=T.Compose([
                                        T.Resize(self.args.image_size, T.InterpolationMode.LANCZOS),
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
                                    download=True)
            test_dataset_per_client = Cifar100FLTest(root=self.args.data_path,
                                                     class_labels=train_dataset.class_labels,
                                                     image_size=self.args.image_size)

        else:
            train_dataset = QuickDrawDatasetFL(root=self.args.data_path,
                                               num_clients=self.args.num_clients,
                                               num_classes=self.args.num_classes,
                                               num_shots=self.args.shots_per_client,
                                               image_size=self.args.image_size,
                                               use_npy=self.args.use_npy_data)

            test_dataset = QuickDrawDatasetFLTest(root=self.args.data_path,
                                                  class_labels=None,
                                                  image_size=self.args.image_size,
                                                  use_npy=self.args.use_npy_data)
            test_dataset_per_client = QuickDrawDatasetFLTest(root=self.args.data_path,
                                                             class_labels=train_dataset.class_labels,
                                                             image_size=self.args.image_size,
                                                             use_npy=self.args.use_npy_data)

        self.model.fed_avg(train_dataset, test_dataset_per_client, test_dataset)

        acc = []

        personalized_film_adapters = []
        for k in range(self.args.num_clients):
            train_dataset.set_client(k)
            train_batch_size = len(train_dataset)
            context_images, context_labels, _ = next(iter(DataLoader(train_dataset,
                                                                     batch_size=train_batch_size,
                                                                     shuffle=True)))
            self.model.fine_tune(context_images, context_labels, iterations=400)
            personalized_film_adapters += [self.model.classifier.get_film_adapter()]
        
            acc += [self.model.test_vtab_fl(test_dataset_per_client, k) * 100]
        print(f'Personalized model lower bound: {np.array(acc).mean():.2f}')

        current_film_state_dict = self.model.classifier.get_film_adapter()

        for key in current_film_state_dict:
            current_film_state_dict[key] = personalized_film_adapters[0][key].clone()/self.args.num_clients
            for j, cur_dict in enumerate(personalized_film_adapters[1:]):
                current_film_state_dict[key] += cur_dict[key].clone()/self.args.num_clients
        self.model.classifier.set_film_adapter(current_film_state_dict, self.model.device)
        self.model._compute_classifier(train_dataset)
        print(f'Oracle model lower bound: {self.model.test_vtab_fl(test_dataset) * 100:.2f}')

        train_dataset.set_all_clients_data(True)
        train_batch_size = len(train_dataset)
        context_images, context_labels, _ = next(iter(DataLoader(train_dataset,
                                                                 batch_size=train_batch_size,
                                                                 shuffle=True)))
        num_iter = 3000 if self.args.dataset == 'quickdraw' else 400
        self.model.fine_tune(context_images, context_labels, iterations=num_iter)
        print(f'Oracle model upper bound: {self.model.test_vtab_fl(test_dataset) * 100:.2f}')

        train_dataset.set_all_clients_data(False)
        acc = []
        for k in range(self.args.num_clients):
            class_labels_idxs = train_dataset.get_client_class_labels(k)
            self.model.classifier.set_classifier_subset(class_labels_idxs)
            acc += [self.model.test_vtab_fl(test_dataset_per_client, k) * 100]
        self.model.classifier.set_all_classes_classifier()
        print(f'Personalized model upper bound: {np.array(acc).mean():.2f}')


if __name__ == "__main__":
    main()
