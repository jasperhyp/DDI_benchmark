from typing import Tuple
import functools
import operator

import torch
import torch.nn as nn
from torchdrug.data import PackedGraph, PackedMolecule

from chemicalx.models import MHCADDI, SSIDDI, Model

from data import ChemicalXBatch

class CustomCASTER(Model):
    """An implementation of the CASTER model from [huang2020]_.

    .. seealso:: This model was suggested in https://github.com/AstraZeneca/chemicalx/issues/17

    .. [huang2020] Huang, K., *et al.* (2020). `CASTER: Predicting drug interactions
       with chemical substructure representation <https://doi.org/10.1609/aaai.v34i01.5412>`_.
       *AAAI 2020 - 34th AAAI Conference on Artificial Intelligence*, 702-709.
    """

    def __init__(
        self,
        *,
        drug_channels: int,
        encoder_hidden_channels: int = 32,
        encoder_output_channels: int = 32,
        decoder_hidden_channels: int = 32,
        hidden_channels: int = 32,
        out_hidden_channels: int = 32,
        out_channels: int = 1,
        lambda3: float = 1e-5,
        magnifying_factor: int = 100,
    ):
        """Instantiate the CASTER model.

        :param drug_channels: The number of drug features (recognised frequent substructures).
            The original implementation recognised 1722 basis substructures in the BIOSNAP experiment.
        :param encoder_hidden_channels: The number of hidden layer neurons in the encoder module.
        :param encoder_output_channels: The number of output layer neurons in the encoder module.
        :param decoder_hidden_channels: The number of hidden layer neurons in the decoder module.
        :param hidden_channels: The number of hidden layer neurons in the predictor module.
        :param out_hidden_channels: The last hidden layer channels before output.
        :param out_channels: The number of output channels.
        :param lambda3: regularisation coefficient in the dictionary encoder module.
        :param magnifying_factor: The magnifying factor coefficient applied to the predictor module input.
        """
        super().__init__()
        self.lambda3 = lambda3
        self.magnifying_factor = magnifying_factor
        self.drug_channels = drug_channels

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.drug_channels, encoder_hidden_channels),
            nn.ReLU(True),
            nn.Linear(encoder_hidden_channels, encoder_output_channels),
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoder_output_channels, decoder_hidden_channels),
            nn.ReLU(True),
            nn.Linear(decoder_hidden_channels, drug_channels),
        )

        # predictor: eight layer NN
        predictor_layers = []
        predictor_layers.append(nn.Linear(self.drug_channels, hidden_channels))
        predictor_layers.append(nn.ReLU(True))
        for i in range(1, 6):
            predictor_layers.append(nn.BatchNorm1d(hidden_channels))
            if i < 5:
                predictor_layers.append(nn.Linear(hidden_channels, hidden_channels))
            else:
                predictor_layers.append(nn.Linear(hidden_channels, out_hidden_channels))
            predictor_layers.append(nn.ReLU(True))
        predictor_layers.append(nn.Linear(out_hidden_channels, out_channels))
        # predictor_layers.append(nn.Sigmoid())
        self.predictor = nn.Sequential(*predictor_layers)
    
    def unpack(self, batch: ChemicalXBatch) -> Tuple[torch.FloatTensor]:
        """Return the "functional representation" of drug pairs, as defined in the original implementation.

        :param batch: batch of drug pairs
        :return: each pair is represented as a single vector with x^i = 1 if either x_1^i >= 1 or x_2^i >= 1
        """
        pair_representation = (torch.maximum(batch.drug_features_left, batch.drug_features_right) >= 1.0).float()
        return (pair_representation,)

    def dictionary_encoder(
        self, drug_pair_features_latent: torch.FloatTensor, dictionary_features_latent: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Perform a forward pass of the dictionary encoder submodule.

        :param drug_pair_features_latent: encoder output for the input drug_pair_features
            (batch_size x encoder_output_channels)
        :param dictionary_features_latent: projection of the drug_pair_features using the dictionary basis
            (encoder_output_channels x drug_channels)
        :return: sparse code X_o: (batch_size x drug_channels)
        """
        dict_feat_squared = torch.matmul(dictionary_features_latent, dictionary_features_latent.transpose(2, 1))
        dict_feat_squared_inv = torch.inverse(dict_feat_squared + self.lambda3 * (torch.eye(self.drug_channels).to(drug_pair_features_latent.device)))
        dict_feat_closed_form = torch.matmul(dict_feat_squared_inv, dictionary_features_latent)
        r = drug_pair_features_latent[:, None, :].matmul(dict_feat_closed_form.transpose(2, 1)).squeeze(1)
        return r

    def forward(self, drug_pair_features: torch.FloatTensor) -> Tuple[torch.FloatTensor, ...]:
        """Run a forward pass of the CASTER model.

        :param drug_pair_features: functional representation of each drug pair (see unpack method)
        :return: (Tuple[torch.FloatTensor): a tuple of tensors including:
                prediction_scores: predicted target scores for each drug pair
                reconstructed: input drug pair vectors reconstructed by the encoder-decoder chain
                dictionary_encoded: drug pair features encoded by the dictionary encoder submodule
                dictionary_features_latent: projection of the encoded drug pair features using the dictionary basis
                drug_pair_features_latent: encoder output for the input drug_pair_features
                drug_pair_features: a copy of the input unpacked drug_pair_features (needed for loss calculation)
        """
        drug_pair_features_latent = self.encoder(drug_pair_features)
        dictionary_features_latent = self.encoder(torch.eye(self.drug_channels).to(drug_pair_features.device))
        dictionary_features_latent = dictionary_features_latent.mul(drug_pair_features[:, :, None])
        drug_pair_features_reconstructed = self.decoder(drug_pair_features_latent)
        reconstructed = torch.sigmoid(drug_pair_features_reconstructed)
        dictionary_encoded = self.dictionary_encoder(drug_pair_features_latent, dictionary_features_latent)
        prediction_scores = self.predictor(self.magnifying_factor * dictionary_encoded)

        return (
            prediction_scores,
            reconstructed,
            dictionary_encoded,
            dictionary_features_latent,
            drug_pair_features_latent,
            drug_pair_features,
        )
      
        
class CustomMHCADDI(MHCADDI):
    def forward(
        self,
        drug_molecules_left: PackedGraph,
        drug_molecules_right: PackedGraph,
    ) -> torch.FloatTensor:
        """Forward pass with the data."""
        outer_segmentation_index_left, outer_index_left, atom_left, bond_left = self._get_molecule_features(
            drug_molecules_left, drug_molecules_right
        )
        outer_segmentation_index_right, outer_index_right, atom_right, bond_right = self._get_molecule_features(
            drug_molecules_right, drug_molecules_left
        )

        drug_left, drug_right = self.encoder(
            drug_molecules_left.node2graph,
            atom_left,
            bond_left,
            drug_molecules_left.edge_list[:, 0],
            drug_molecules_left.edge_list[:, 1],
            outer_segmentation_index_left,
            outer_index_left,
            drug_molecules_right.node2graph,
            atom_right,
            bond_right,
            drug_molecules_right.edge_list[:, 0],
            drug_molecules_right.edge_list[:, 1],
            outer_segmentation_index_right,
            outer_index_right,
        )

        # prediction_left = self.head_layer(torch.cat([drug_left, drug_right], dim=1))
        # prediction_right = self.head_layer(torch.cat([drug_right, drug_left], dim=1))
        # prediction_mean = (prediction_left + prediction_right) / 2
        
        predictions = self.head_layer(torch.cat([drug_left, drug_right], dim=1))  # NOTE: we already make the edgelist bidirectional during dat a loading
        
        # return torch.sigmoid(prediction_mean)
        return predictions
    
    def generate_outer_segmentation(self, graph_sizes_left: torch.LongTensor, graph_sizes_right: torch.LongTensor):
        """Calculate all pairwise edges between the atoms in a set of drug pairs.

        Example: Given two sets of drug sizes:

        graph_sizes_left = torch.tensor([1, 2])
        graph_sizes_right = torch.tensor([3, 4])

        Here the drug pairs have sizes (1,3) and (2,4)

        This results in:

        outer_segmentation_index = tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        outer_index = tensor([0, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6])

        :param graph_sizes_left: List of graph sizes in the left drug batch.
        :param graph_sizes_right: List of graph sizes in the right drug batch.
        :returns: Edge indices.
        """
        interactions = graph_sizes_left * graph_sizes_right

        left_shifted_graph_size_cum_sum = torch.cumsum(graph_sizes_left, 0) - graph_sizes_left
        shift_sums_left = torch.repeat_interleave(left_shifted_graph_size_cum_sum, interactions)
        outer_segmentation_index = [
            torch.repeat_interleave(torch.arange(left_graph_size.item()), right_graph_size.item())
            for left_graph_size, right_graph_size in zip(graph_sizes_left, graph_sizes_right)
        ]
        outer_segmentation_index = functools.reduce(operator.iconcat, outer_segmentation_index, [])
        outer_segmentation_index = torch.tensor(outer_segmentation_index).to(shift_sums_left.device) + shift_sums_left

        right_shifted_graph_size_cum_sum = torch.cumsum(graph_sizes_right, 0) - graph_sizes_right
        shift_sums_right = torch.repeat_interleave(right_shifted_graph_size_cum_sum, interactions)
        outer_index = [
            list(range(0, right_graph_size.item())) * left_graph_size.item()
            for left_graph_size, right_graph_size in zip(graph_sizes_left, graph_sizes_right)
        ]
        outer_index = functools.reduce(operator.iconcat, outer_index, [])
        outer_index = torch.tensor(outer_index).to(shift_sums_right.device) + shift_sums_right
        return outer_segmentation_index, outer_index
        

class CustomEmbeddingLayer(nn.Module):
    """Attention layer."""

    def __init__(self, feature_number: int, output_channels: int):
        """Initialize the relational embedding layer.

        :param feature_number: Number of features.
        """
        super().__init__()
        self.output_channels = output_channels
        self.weights = nn.Parameter(torch.zeros(output_channels, feature_number, feature_number))
        nn.init.xavier_uniform_(self.weights)

    def forward(
        self,
        left_representations: torch.FloatTensor,
        right_representations: torch.FloatTensor,
        alpha_scores: torch.FloatTensor,
    ):
        """
        Make a forward pass with the drug representations.

        :param left_representations: Left side drug representations.
        :param right_representations: Right side drug representations.
        :param alpha_scores: Attention scores.
        :returns: Positive label scores vector.
        """
        attention = nn.functional.normalize(self.weights, dim=-1)
        left_representations = nn.functional.normalize(left_representations, dim=-1)
        right_representations = nn.functional.normalize(right_representations, dim=-1)
        # attention = attention.view(-1, self.weights.shape[1], self.weights.shape[2])
        
        # FIXME: FIX THIS TO MULTILABEL
        # scores = alpha_scores * (left_representations @ attention @ right_representations.transpose(-2, -1))
        left_repr_expanded = left_representations.unsqueeze(1)
        right_repr_expanded = right_representations.unsqueeze(1).transpose(-2, -1)
        scores = alpha_scores.unsqueeze(1) * torch.matmul(torch.matmul(left_repr_expanded, attention), right_repr_expanded)
        scores = scores.sum(dim=(-2, -1)).view(-1, self.output_channels)
        return scores
        
        
class CustomSSIDDI(SSIDDI):
    def __init__(self, molecule_channels, hidden_channels, head_number, output_channels):
        super(CustomSSIDDI, self).__init__(molecule_channels=molecule_channels, hidden_channels=hidden_channels, head_number=head_number)
        self.relational_embedding = CustomEmbeddingLayer(hidden_channels[-1], output_channels)
    
    def _forward_molecules(self, molecules: PackedGraph):
        molecules.node_feature = self.initial_norm(molecules.node_feature.float())
        representation = []
        for block, net_norm in zip(self.blocks, self.net_norms):
            molecules, pooled_hidden_left = block(molecules)
            representation.append(pooled_hidden_left)
            molecules.node_feature = torch.nn.functional.elu(net_norm(molecules.node_feature))
        return torch.stack(representation, dim=-2)
    
    def forward(self, molecules_left: PackedGraph, molecules_right: PackedGraph) -> torch.FloatTensor:
        """Run a forward pass of the SSI-DDI model.

        :param molecules_left: Batched molecules for the left side drugs.
        :param molecules_right: Batched molecules for the right side drugs.
        :returns: A column vector of predicted synergy scores.
        """
        features_left = self._forward_molecules(molecules_left)
        features_right = self._forward_molecules(molecules_right)
        hidden = self._combine_sides(features_left, features_right)
        # return self.final(hidden)
        return hidden

