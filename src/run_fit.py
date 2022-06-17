# Note:
# Throughout this code, we use the nomenclature of context set and target set instead of
# support set and query set, respectively, that is used in the paper.

import os.path
import torch
import argparse
from dataset import vtab_datasets, few_shot_datasets
from utils import Logger, compute_accuracy, limit_tensorflow_memory_usage, CsvWriter, get_mean_percent
from model import FiT
from tf_dataset_reader import TfDatasetReader
from datetime import datetime


def main():
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()
        self.logger = Logger(self.args.checkpoint_dir, 'log.txt')

        self.logger.print_and_log("Start Time = {}".format(datetime.now()))
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

        parser.add_argument('--mode', help='Mode for testing', choices=["vtab_1000", "few_shot"],
                            default="vtab_1000")
        parser.add_argument("--feature_extractor", choices=["efficientnet-b0", 'BiT-M-R50x1-FILM'],
                            default='BiT-M-R50x1-FILM', help="Feature extractor to use.")
        parser.add_argument("--classifier", choices=['protonets', 'lda', 'qda'], default="lda",
                            help="Classifier to use.")
        parser.add_argument("--data_path", default="../datasets", help="Path to dataset root folder.")
        parser.add_argument("--download_path_for_tensorflow_datasets", default=None,
                            help="Path to download the tensorflow datasets.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.0035, help="Learning rate.")
        parser.add_argument("--checkpoint_dir", "-c", default='../checkpoints',
                            help="Directory to save checkpoint to.")
        parser.add_argument("--iterations", "-i", type=int, default=400, help="Number of fine-tune iterations.")
        parser.add_argument("--train_batch_size", "-b", type=int, default=100, help="Batch size.")
        parser.add_argument("--test_batch_size", "-tb", type=int, default=600, help="Batch size.")
        parser.add_argument("--examples_per_class", type=int, default=None,
                            help="Examples per class when doing few-shot. -1 indicates to use the entire training set.")
        parser.add_argument("--examples_per_class_seed", type=int, default=0,
                            help="Seed for VTAB fixed shot sampler.")
        parser.add_argument("--tfds_seed", type=int, default=0, help="Seed for tensorflow datasets.")
        parser.add_argument("--save_model", dest="save_model", default=False,
                            action="store_true", help="If true, save the fine tuned model.")
        parser.add_argument("--do_not_split", dest="do_not_split",
                            default=False, action="store_true", help="If true, do not split the context set.")
        parser.add_argument("--do_not_use_tasks", dest="do_not_use_tasks", default=False, action="store_true",
                            help="If true, do not create small tasks (i.e. use all data).")

        args = parser.parse_args()

        return args

    def run(self):
        limit_tensorflow_memory_usage(1024)
        self.test()
        self.logger.print_and_log("Finish Time = {}".format(datetime.now()))

    def test(self):
        self.logger.print_and_log("")  # add a blank line

        if self.args.mode == "few_shot":
            datasets = few_shot_datasets
            summary_dict = None
        else:
            datasets = vtab_datasets
            summary_dict = {
                'all': [],
                'natural': [],
                'specialized': [],
                'structured': [],
            }

        with torch.no_grad():
            for dataset in datasets:
                if dataset['enabled'] is False:
                    continue

                if self.args.examples_per_class == -1:
                    context_set_size = -1  # this is the use the entire training set case
                elif (self.args.examples_per_class is not None) and (dataset['name'] != 'oxford_iiit_pet'):  # bug in pets
                    context_set_size = self.args.examples_per_class * dataset['num_classes']  # few-shot case
                else:
                    context_set_size = 1000  # VTAB1000

                dataset_reader = TfDatasetReader(
                    dataset=dataset['name'],
                    task=dataset['task'],
                    context_batch_size=context_set_size,
                    target_batch_size=self.args.test_batch_size,
                    path_to_datasets=self.args.download_path_for_tensorflow_datasets,
                    num_classes=dataset['num_classes'],
                    image_size=dataset['image_size'],
                    examples_per_class=self.args.examples_per_class if self.args.examples_per_class != -1 else None,
                    examples_per_class_seed=self.args.examples_per_class_seed,
                    tfds_seed=self.args.tfds_seed,
                    device=self.device,
                    osr=False
                )
                context_images, context_labels = dataset_reader.get_context_batch()

                # fine tune the model to the current task
                self.model.fine_tune(context_images, context_labels)

                # test the model
                accuracy = (self.model.test_vtab(dataset_reader, context_labels, dataset['num_classes'])).cpu()
                if summary_dict is not None:
                    summary_dict['all'].append(accuracy)
                    summary_dict[dataset['category']].append(accuracy)
                if dataset['task'] is None:
                    self.logger.print_and_log('{0:}: {1:3.1f}'.format(dataset['name'], accuracy * 100.0))
                else:
                    self.logger.print_and_log('{0:} {1:}: {2:3.1f}'.format(dataset['name'], dataset['task'], accuracy * 100.0))

                self.csv_writer.write_row(
                    [
                        "{0:}".format(dataset['name']),
                        "{0:3.1f}".format(accuracy * 100.00)
                    ]
                )

                # save out the model
                if self.args.save_model:
                    self.model.classifier.save(os.path.join(self.args.checkpoint_dir, dataset['model_name'] + '.pt'))

            if summary_dict is not None:
                for key in summary_dict:
                    acc = get_mean_percent(summary_dict[key])
                    self.logger.print_and_log("{0}: {1:3.1f}".format(key, acc))
                    self.csv_writer.write_row(
                        [
                            "{0:}".format(key),
                            "{0:3.1f}".format(acc)
                        ]
                    )


if __name__ == "__main__":
    main()
