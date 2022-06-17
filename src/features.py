import torch
import torch.nn as nn
import numpy as np
from efficientnet import film_efficientnet
from bit_resnet import KNOWN_MODELS


def create_feature_extractor(args):
    if "efficientnet" in args.feature_extractor:
        feature_extractor = film_efficientnet(args.feature_extractor)
    else:
        feature_extractor = KNOWN_MODELS[args.feature_extractor]()
        weights_model_name = (args.feature_extractor).replace("-FILM", "")
        feature_extractor.load_from(np.load(f"{weights_model_name}.npz"))
        print("Backbone Parameter Count = {}".format(feature_extractor.get_parameter_count()))

    # freeze the parameters of feature extractor
    for param in feature_extractor.parameters():
        param.requires_grad = False

    return feature_extractor


def create_film_adapter(feature_extractor):
    adaptation_layer = FilmLayer
    adaptation_config = feature_extractor.get_adaptation_config()
    feature_adapter = FilmAdapter(
        layer=adaptation_layer,
        adaptation_config=adaptation_config
    )

    return feature_adapter


class BaseFilmLayer(nn.Module):
    def __init__(self, num_maps, num_blocks):
        super(BaseFilmLayer, self).__init__()

        self.num_maps = num_maps
        self.num_blocks = num_blocks
        self.num_generated_params = 0


class FilmLayer(BaseFilmLayer):
    def __init__(self, num_maps, num_blocks, task_dim=None):
        BaseFilmLayer.__init__(self, num_maps, num_blocks)

        self.gammas = nn.ParameterList()
        self.betas = nn.ParameterList()

        for i in range(self.num_blocks):
            self.gammas.append(nn.Parameter(torch.ones(self.num_maps[i]), requires_grad=True))
            self.betas.append(nn.Parameter(torch.zeros(self.num_maps[i]), requires_grad=True))

    def forward(self, x):
        block_params = []
        for block in range(self.num_blocks):
            block_param_dict = {
                'gamma': self.gammas[block],
                'beta': self.betas[block]
            }
            block_params.append(block_param_dict)
        return block_params


class FilmAdapter(nn.Module):
    def __init__(self, layer, adaptation_config, task_dim=None):
        super().__init__()
        self.num_maps = adaptation_config['num_maps_per_layer']
        self.num_blocks = adaptation_config['num_blocks_per_layer']
        self.task_dim = task_dim
        self.num_target_layers = len(self.num_maps)
        self.layer = layer
        self.num_generated_params = 0
        self.layers = self.get_layers()

    def get_layers(self):
        layers = nn.ModuleList()
        for num_maps, num_blocks in zip(self.num_maps, self.num_blocks):
            layers.append(
                self.layer(
                    num_maps=num_maps,
                    num_blocks=num_blocks,
                    task_dim=self.task_dim
                )
            )
            self.num_generated_params += layers[-1].num_generated_params
        return layers

    def forward(self, x):
        return [self.layers[layer](x) for layer in range(self.num_target_layers)]


