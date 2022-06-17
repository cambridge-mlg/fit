import torch
import numpy as np
from utils import cross_entropy_loss, compute_accuracy, shuffle, predict_by_max_logit, compute_accuracy_from_predictions
from classifier import NaiveBayesClassifier
from features import create_feature_extractor
from dataset import TaskResampler
from torch.utils.data import DataLoader


class FiT:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.classifier = None
        self.feature_extractor = create_feature_extractor(args).to(device)
        self.loss = cross_entropy_loss
        self.accuracy = compute_accuracy

    def fed_avg(self, train_dataset, test_dataset_per_client, test_dataset):
        self.classifier = None
        self.classifier = NaiveBayesClassifier(
            feature_extractor=self.feature_extractor,
            args=self.args
        ).to(self.device)

        self.classifier.feature_extractor.eval()

        current_film_state_dict = self.classifier.get_film_adapter()

        num_clients_per_epoch = self.args.num_clients_per_round

        # initial accuracy
        with torch.no_grad():

            self._compute_classifier(train_dataset)
            print(f'Initial global accuracy: acc {self.test_vtab_fl(test_dataset)*100:.2f}')

            acc = []
            for k in range(self.args.num_clients):
                train_dataset.set_client(k)
                train_batch_size = len(train_dataset)
                context_images, context_labels, _ = next(iter(DataLoader(train_dataset,
                                                                         batch_size=train_batch_size,
                                                                         shuffle=True)))
                context_images = context_images.to(self.device)
                context_labels = context_labels.to(self.device)
                context_features = self._compute_features_by_batch(context_images, self.args.test_batch_size)
                self.classifier.configure_from_features(context_features, context_labels)
                acc += [self.test_vtab_fl(test_dataset_per_client, k) * 100]
            print(f'Initial personalized accuracy: acc {np.array(acc).mean():.2f}')

        lr_scale = 1.0

        for iteration in range(self.args.iterations):
            current_state_dicts = []
            current_clients = np.random.choice(self.args.num_clients, num_clients_per_epoch, replace=False)

            if iteration % 20 == 0 and iteration > 0 and self.args.dataset == 'cifar100':
               lr_scale *= 0.3

            datasets_length = []

            for k in current_clients:
                train_dataset.set_client(k)
                train_batch_size = len(train_dataset)

                datasets_length += [len(train_dataset)]
                context_images, context_labels, _ = next(iter(DataLoader(train_dataset,
                                                                      batch_size=train_batch_size,
                                                                      shuffle=True)))

                task_resampler = TaskResampler(context_images,
                                               context_labels,
                                               self.args.train_batch_size,
                                               device=self.device)
                self.classifier.set_film_adapter(current_film_state_dict, self.device)
                optimizer = torch.optim.Adam(self.classifier.parameters(),
                                             lr=self.args.learning_rate*lr_scale,
                                             weight_decay=self.args.weight_decay)
                optimizer.zero_grad()
                for _ in range(self.args.num_local_epochs):
                    batch_context_images, batch_context_labels, batch_target_images, batch_target_labels = task_resampler.get_task()

                    batch_target_set_size = len(batch_target_labels)
                    num_batches = int(np.ceil(float(batch_target_set_size) / float(self.args.train_batch_size)))
                    if num_batches == 0:
                        continue
                    for batch in range(num_batches):
                        self.classifier.configure_from_images(batch_context_images, batch_context_labels)
                        batch_start_index, batch_end_index = self._get_batch_indices(batch, batch_target_set_size,
                                                                                     self.args.train_batch_size)
                        batch_len = len(batch_target_labels[batch_start_index:batch_end_index])
                        logits = self.classifier.predict(batch_target_images[batch_start_index:batch_end_index])
                        loss = self.loss(logits, batch_target_labels[batch_start_index:batch_end_index])*batch_len/batch_target_set_size
                        loss.backward()
                        del logits
                    optimizer.step()
                    optimizer.zero_grad()

                current_state_dicts += [self.classifier.get_film_adapter()]

            all_data_len = np.array(datasets_length).sum()
            for key in current_film_state_dict:
                current_film_state_dict[key] = datasets_length[0]/all_data_len*current_state_dicts[0][key].clone()
                for j, cur_dict in enumerate(current_state_dicts[1:]):
                    current_film_state_dict[key] += datasets_length[j]/all_data_len*cur_dict[key].clone()
            self.classifier.set_film_adapter(current_film_state_dict, self.device)

            if iteration % 10 == 0 or iteration == self.args.iterations - 1:

                with torch.no_grad():
                    acc = []
                    for k in range(self.args.num_clients):
                        train_dataset.set_client(k)
                        train_batch_size = len(train_dataset)
                        context_images, context_labels, _ = next(iter(DataLoader(train_dataset,
                                                                                 batch_size=train_batch_size,
                                                                                 shuffle=True)))
                        context_images = context_images.to(self.device)
                        context_labels = context_labels.to(self.device)
                        context_features = self._compute_features_by_batch(context_images, self.args.test_batch_size)
                        self.classifier.configure_from_features(context_features, context_labels)
                        acc += [self.test_vtab_fl(test_dataset_per_client, k)*100]
                    print(f'Current iteration {iteration}: acc {np.array(acc).mean():.2f}')

                    self._compute_classifier(train_dataset)
                    print(f'Current iteration {iteration}: global acc {self.test_vtab_fl(test_dataset) * 100:.2f}')

    def _compute_classifier(self, train_dataset):
        # not the best way to create classifier for federated learning,
        # but it would be the same as averaging clients' classifiers, as our clients have datasets of equal size.
        with torch.no_grad():
            all_context_features = []
            all_context_labels = []

            for k in range(self.args.num_clients):
                train_dataset.set_client(k)
                train_batch_size = len(train_dataset)
                context_images, _, context_labels = next(iter(DataLoader(train_dataset,
                                                                         batch_size=train_batch_size,
                                                                         shuffle=True)))
                context_images = context_images.to(self.device)
                context_labels = context_labels.to(self.device)
                all_context_features += [self._compute_features_by_batch(context_images, self.args.test_batch_size)]
                all_context_labels += [context_labels]
            all_context_features = torch.cat(all_context_features, dim=0)
            all_context_labels = torch.cat(all_context_labels, dim=0)
            self.classifier.configure_from_features(all_context_features, all_context_labels)

    def fine_tune(self, context_images, context_labels, iterations=None):
        self.classifier = None
        self.classifier = NaiveBayesClassifier(
            feature_extractor=self.feature_extractor,
            args=self.args
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.args.learning_rate)

        self.classifier.feature_extractor.eval()

        context_images, context_labels = shuffle(context_images, context_labels)
        if not self.args.do_not_use_tasks:
            if self.args.do_not_split:
                task_resampler = TaskResampler(context_images, context_labels, self.args.train_batch_size,
                                               device=self.device, target_images=context_images, target_labels=context_labels)
            else:
                task_resampler = TaskResampler(context_images, context_labels, self.args.train_batch_size, device=self.device)

        if iterations is None:
            iterations = self.args.iterations

        for iteration in range(iterations):
            if not self.args.do_not_use_tasks:
                batch_context_images, batch_context_labels, batch_target_images, batch_target_labels = \
                    task_resampler.get_task()

            if self.args.do_not_use_tasks:
                torch.set_grad_enabled(False)
                context_images = context_images.to(self.device)
                context_labels = context_labels.to(self.device)
                batch_target_images = context_images
                batch_target_labels = context_labels
                context_features = self._compute_features_by_batch(context_images, self.args.test_batch_size)
                self.classifier.configure_from_features(context_features, context_labels)

            batch_target_set_size = len(batch_target_labels)
            num_batches = int(np.ceil(float(batch_target_set_size) / float(self.args.train_batch_size)))
            if num_batches == 0:
                # TODO: This occurs when there is only one image per class
                #  and since there is no augmentation, we have missing targets.
                #  Fix this by adding a flipped version of the image in the target set.
                #  This likely only happens in MD, as VTAB context sets are larger.
                continue
            optimizer.zero_grad()
            torch.set_grad_enabled(True)
            for batch in range(num_batches):
                if not self.args.do_not_use_tasks:
                    self.classifier.configure_from_images(batch_context_images, batch_context_labels)

                batch_start_index, batch_end_index = self._get_batch_indices(batch, batch_target_set_size,
                                                                             self.args.train_batch_size)
                logits = self.classifier.predict(batch_target_images[batch_start_index:batch_end_index])
                batch_len = len(batch_target_labels[batch_start_index:batch_end_index])
                loss = self.loss(logits, batch_target_labels[batch_start_index:batch_end_index])*batch_len/batch_target_set_size
                loss.backward()
                del logits
            optimizer.step()
        with torch.no_grad():
            context_images = context_images.to(self.device)
            context_labels = context_labels.to(self.device)
            context_features = self._compute_features_by_batch(context_images, self.args.test_batch_size)
            self.classifier.configure_from_features(context_features, context_labels)

    def _compute_features_by_batch(self, images, batch_size):
        features = []
        num_images = images.size(0)
        num_batches = int(np.ceil(float(num_images) / float(batch_size)))
        for batch in range(num_batches):
            batch_start_index, batch_end_index = self._get_batch_indices(batch, num_images, batch_size)
            film_params = self.classifier.film_adapter(None)
            features.append(self.classifier.feature_extractor(images[batch_start_index: batch_end_index], film_params))
        return torch.vstack(features)

    def test_vtab(self, dataset_reader, context_labels=None, num_classes = None):
        self.classifier.feature_extractor.eval()
        test_set_size = dataset_reader.get_target_dataset_length()
        num_batches = int(np.ceil(float(test_set_size) / float(self.args.test_batch_size)))

        with torch.no_grad():
            labels = []
            predictions = []
            for batch in range(num_batches):
                batch_images, batch_labels = dataset_reader.get_target_batch()
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.type(torch.LongTensor).to(self.device)
                logits = self.classifier.predict(batch_images)
                predictions.append(predict_by_max_logit(logits))
                labels.append(batch_labels)
                del logits
            predictions = torch.hstack(predictions)
            if (context_labels is not None) and (num_classes != len(torch.unique(context_labels))):
                # This can happen when the context set does not contain all the classes in the dataset.
                # In particular for SUN397 in VTAB-1k, in which the values for some classes are rare and
                # may not get sampled.
                predictions = self.adjust_predictions(predictions, context_labels)

            labels = torch.hstack(labels)
            accuracy = compute_accuracy_from_predictions(predictions, labels)
        return accuracy

    def adjust_predictions(self, predictions, context_labels):
        actual_labels = torch.unique(context_labels)
        return actual_labels[predictions].to(self.device)

    def test_vtab_fl(self, test_dataset, k=None):
        self.classifier.feature_extractor.eval()

        with torch.no_grad():
            if k is not None:
                test_dataset.set_client(k)
                test_dataloader = DataLoader(test_dataset, batch_size=self.args.test_batch_size)
                labels = []
                predictions = []
                for batch_images, batch_labels, _ in test_dataloader:
                    batch_images = batch_images.to(self.device)
                    batch_labels = batch_labels.type(torch.LongTensor).to(self.device)
                    logits = self.classifier.predict(batch_images)
                    predictions.append(predict_by_max_logit(logits))
                    labels.append(batch_labels)
                    del logits
                predictions = torch.hstack(predictions)
                labels = torch.hstack(labels)
                accuracy = compute_accuracy_from_predictions(predictions, labels).item()
            else:
                test_dataloader = DataLoader(test_dataset, batch_size=self.args.test_batch_size, num_workers=4)
                labels = []
                predictions = []
                for batch_images, batch_labels in test_dataloader:
                    batch_images = batch_images.to(self.device)
                    batch_labels = batch_labels.type(torch.LongTensor).to(self.device)
                    logits = self.classifier.predict(batch_images)
                    predictions.append(predict_by_max_logit(logits))
                    labels.append(batch_labels)
                    del logits
                predictions = torch.hstack(predictions)
                labels = torch.hstack(labels)
                accuracy = compute_accuracy_from_predictions(predictions, labels).item()
        return accuracy

    def _get_batch_indices(self, index, last_element, batch_size):
        batch_start_index = index * batch_size
        batch_end_index = batch_start_index + batch_size
        if batch_end_index > last_element:
            batch_end_index = last_element
        return batch_start_index, batch_end_index
