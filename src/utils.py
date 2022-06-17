import os
import torch
import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import csv


class Logger():
    def __init__(self, checkpoint_dir, log_file_name):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        log_file_path = os.path.join(checkpoint_dir, log_file_name)
        self.file = None
        if os.path.isfile(log_file_path):
            self.file = open(log_file_path, "a", buffering=1)
        else:
            self.file = open(log_file_path, "w", buffering=1)

    def __del__(self):
        self.file.close()

    def log(self, message):
        self.file.write(message + '\n')

    def print_and_log(self, message):
        print(message, flush=True)
        self.log(message)


def compute_accuracy(logits, labels):
    """
    Compute classification accuracy.
    """
    return torch.mean(torch.eq(labels, torch.argmax(logits, dim=-1)).float())


def cross_entropy_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def shuffle(images, labels):
    """
    Return shuffled data.
    """
    permutation = np.random.permutation(images.shape[0])
    return images[permutation], labels[permutation]


def set_pytorch_seeds():
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict_by_max_logit(logits):
    return torch.argmax(logits, dim=-1)


def compute_accuracy_from_predictions(predictions, labels):
    """
    Compute classification accuracy.
    """
    return torch.mean(torch.eq(labels, predictions).float())


def limit_tensorflow_memory_usage(gpu_memory_limit):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit)]
                )
        except RuntimeError as e:
            print(e)


class CsvWriter:
    def __init__(self, file_path, header):
        self.file = open(file_path, 'w', encoding='UTF8', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)

    def __del__(self):
        self.file.close()

    def write_row(self, row):
        self.writer.writerow(row)


def get_mean_percent_and_95_confidence_interval(list_of_quantities):
    if len(list_of_quantities) > 1:
        mean_percent = np.array(list_of_quantities).mean() * 100.0
        confidence_interval = (196.0 * np.array(list_of_quantities).std()) / np.sqrt(len(list_of_quantities))
        return mean_percent, confidence_interval
    else:
        return list_of_quantities[0] * 100.0, 0.0


def get_mean_percent(list_of_quantities):
    mean_percent = np.array(list_of_quantities).mean() * 100.0
    return mean_percent


def extract_class_indices(labels, which_class):
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector
