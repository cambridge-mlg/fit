import torch
import torch.nn as nn
from features import create_film_adapter
from naive_bayes import NaiveBayesPredictor
import sys


class NaiveBayesClassifier(nn.Module):
    def __init__(self, feature_extractor, args):
        super(NaiveBayesClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.film_adapter = create_film_adapter(self.feature_extractor)
        self.predictor = NaiveBayesPredictor(args)
        self.means = None
        self.trils = None
        self.all_means = None
        if (args.classifier == "lda") or (args.classifier == "qda"):
            self.predict_fn = self.predictor.predict_naive_bayes
        elif args.classifier == "protonets":
            self.predict_fn = self.predictor.predict_protonets
        else:
            print("Invalid classifier option.")
            sys.exit()

    def configure_from_images(self, context_images, context_labels):
        film_params = self.film_adapter(None)
        context_features = self.feature_extractor(context_images, film_params)
        self.configure_from_features(context_features, context_labels)

    def set_classifier_subset(self, subset_idxs):
        if self.all_means is None:
            self.all_means = self.means.clone()
        self.means = self.all_means[subset_idxs]

    def set_all_classes_classifier(self):
        self.means = self.all_means.clone()
        self.all_means = None

    def configure_from_features(self, context_features, context_labels):
        self.means, self.trils = self.predictor.compute_class_means_and_trils(context_features, context_labels)

    def predict(self, target_images):
        film_params = self.film_adapter(None)
        target_features = self.feature_extractor(target_images, film_params)
        return self.predict_fn(target_features, self.means, self.trils)

    def save(self, path):
        # save only the FiLM and cov reg params
        torch.save({
            'predictor_state_dict': self.predictor.state_dict(),
            'film_state_dict': self.film_adapter.state_dict(),
            'means': self.means,
            'trils': self.trils
         }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        self.film_adapter.load_state_dict(checkpoint['film_state_dict'])
        self.means = checkpoint['means']
        self.trils = checkpoint['trils']

    def set_film_adapter(self, film_adapter_state_dict, device):
        self.film_adapter = create_film_adapter(self.feature_extractor).to(device)
        self.film_adapter.load_state_dict(film_adapter_state_dict)

    def get_film_adapter(self):
        return self.film_adapter.state_dict()
