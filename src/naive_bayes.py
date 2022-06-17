import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import multivariate_normal
from utils import extract_class_indices


class NaiveBayesPredictor(nn.Module):
    def __init__(self, args):
        super(NaiveBayesPredictor, self).__init__()
        self.args = args
        if self.args.classifier == "qda":
            self.class_cov_weight = nn.Parameter(torch.tensor([0.5]))
            self.task_cov_weight = nn.Parameter(torch.tensor([0.5]))
            self.cov_reg_weight = nn.Parameter(torch.tensor([1.0]))
        elif self.args.classifier == "lda":
            self.task_cov_weight = nn.Parameter(torch.tensor([1.0]))
            self.cov_reg_weight = nn.Parameter(torch.tensor([1.0]))
        return

    def predict_naive_bayes(self, target_features, class_means, class_trils):
        class_distributions = multivariate_normal.MultivariateNormal(
            loc=class_means,
            scale_tril=class_trils
        )

        return class_distributions.log_prob(target_features.unsqueeze(dim=1))  # (num_targets, num_classes)

    def predict_protonets(self, target_features, class_means, class_trils):
        num_target_features = target_features.shape[0]
        num_prototypes = class_means.shape[0]

        distances = (target_features.unsqueeze(1).expand(num_target_features, num_prototypes, -1) -
                    class_means.unsqueeze(0).expand(num_target_features, num_prototypes, -1)).pow(2).sum(dim=2)
        return -distances       

    def compute_class_means_and_trils(self, features, labels):
        means = []
        trils = []

        # need task cov for qda and lda, but not protonets
        if self.args.classifier != "protonets":
            task_covariance_estimate = self._estimate_cov(features)

        # lda just needs the task cov
        if self.args.classifier == "lda":
            trils.append(self._lower_triangular(
                F.relu(self.task_cov_weight) * task_covariance_estimate +
                F.relu(self.cov_reg_weight) * torch.eye(features.size(1), features.size(1)).cuda(0)))

        for c in torch.unique(labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(features, 0, extract_class_indices(labels, c))
            # mean pooling examples to form class means
            means.append(self._mean_pooling(class_features).squeeze())
            # compute class cov for qda
            if self.args.classifier == "qda":
                trils.append(self._lower_triangular(
                    F.relu(self.class_cov_weight) * self._estimate_cov(class_features) +
                    F.relu(self.task_cov_weight) * task_covariance_estimate +
                    F.relu(self.cov_reg_weight) * torch.eye(class_features.size(1), class_features.size(1)).cuda(0)))

        means = (torch.stack(means))
        if self.args.classifier != "protonets":
            trils = (torch.stack(trils))

        return means, trils

    @staticmethod
    def _estimate_cov(examples):
        if examples.size(0) > 1:
            return torch.cov(examples.t(), correction=1)
        else:
            factor = 1.0 / (examples.size(1) - 1)
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
            return factor * examples.matmul(examples.t()).squeeze()

    @staticmethod
    def _lower_triangular(matrix):
        return torch.linalg.cholesky(matrix)

    @staticmethod
    def _mean_pooling(x):
        return torch.mean(x, dim=0, keepdim=True)


