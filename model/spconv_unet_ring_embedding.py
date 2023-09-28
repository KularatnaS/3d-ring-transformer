import torch
import torch.nn as nn

import functools
from collections import OrderedDict

from einops import rearrange

import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, norm_fn, indice_key=None):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        self.conv_branch = spconv.SparseSequential(
            norm_fn(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            norm_fn(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)

        output = output.replace_feature(output.features + self.i_branch(identity).features)

        return output


class UBlock(nn.Module):
    def __init__(self, nPlanes, norm_fn, block_reps, block, indice_key_id=1):

        super().__init__()

        self.nPlanes = nPlanes

        blocks = {'block{}'.format(i): block(nPlanes[0], nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id)) for i in range(block_reps)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            self.conv = spconv.SparseSequential(
                norm_fn(nPlanes[0]),
                nn.ReLU(),
                spconv.SparseConv3d(nPlanes[0], nPlanes[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            self.u = UBlock(nPlanes[1:], norm_fn, block_reps, block, indice_key_id=indice_key_id+1)

            self.deconv = spconv.SparseSequential(
                norm_fn(nPlanes[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail['block{}'.format(i)] = block(nPlanes[0] * (2 - i), nPlanes[0], norm_fn, indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output_features = torch.cat((identity.features, output_decoder.features), dim=1)
            output = output.replace_feature(output_features)

            output = self.blocks_tail(output)

        return output


class RingEmbedding(nn.Module):
    def __init__(self, n_point_features, n_extracted_point_features, model_resolution, d_ring_embedding):
        super().__init__()
        self.model_resolution = model_resolution

        input_c = n_point_features
        m = n_extracted_point_features
        block_reps = 2
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block = ResidualBlock

        # backbone
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_c, m, kernel_size=3, padding=1, bias=False, indice_key='subm1')
        )

        self.unet = UBlock([m, 2*m, 3*m], norm_fn, block_reps, block, indice_key_id=1)

        # per point features
        self.output_layer = spconv.SparseSequential(
            norm_fn(m),
            nn.ReLU()
        )

        # ring embedding
        self.linear = nn.Linear(m, d_ring_embedding)

    def forward(self, x):
        # x -> [batch, n_rings, n_points_per_ring, n_point_features]
        input_batch_size = x.shape[0]
        input_n_rings = x.shape[1]
        input_n_points_per_ring = x.shape[2]

        features, indices, spatial_shape, batch_size = batch_to_spconv_tensor(x, self.model_resolution)

        x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
        x = self.input_conv(x)
        x = self.unet(x)

        # per point features
        x = self.output_layer(x)  # [batch * n_rings * n_points_per_ring, n_extracted_point_features]
        per_point_embedded_features = rearrange(x.features, '(batch n_rings n_points)  f -> batch n_rings n_points f',
                                                batch=input_batch_size, n_rings=input_n_rings,
                                                n_points=input_n_points_per_ring)

        x = self.linear(x.features)  # [batch * n_rings * n_points_per_ring, d_ring_embedding]
        # reshape to [batch * n_rings, d_ring_embedding, n_points_per_ring]
        x = rearrange(x, '(batch n_rings n_points)  d -> (batch n_rings) d n_points',
                                                batch=input_batch_size, n_rings=input_n_rings,
                                                n_points=input_n_points_per_ring)
        x = torch.max(x, 2, keepdim=True)[0]  # [batch * n_rings, d_ring_embedding, 1]
        x = rearrange(x, 'a b 1 -> a b')  # [batch * n_rings, d_ring_embedding]
        x = rearrange(x, '(batch n_rings) b -> batch n_rings b', batch=input_batch_size,
                      n_rings=input_n_rings)  # [batch, n_rings, d_ring_embedding]

        return x, per_point_embedded_features


def batch_to_spconv_tensor(x, model_resolution):
    device = x.device
    #model_resolution = torch.tensor(model_resolution).to(device)

    # x -> [batch, n_rings, n_points_per_ring, n_point_features]
    batch_size = x.shape[0] * x.shape[1]
    batch_size = torch.tensor(batch_size).to(torch.int32)

    features = rearrange(x, 'batch a b c -> (batch a b) c')  # [batch * n_rings * n_points_per_ring, n_point_features]

    indices = (features - features.min(0).values) * (1/model_resolution)
    indices = indices.int()
    batch_indices = torch.arange(batch_size).repeat_interleave(x.shape[2]).reshape(-1, 1).to(device)
    indices = torch.cat((batch_indices, indices), dim=1).to(torch.int32)

    # get max of all columns of indices except for the first column
    spatial_shape = indices[:, 1:].max(0).values + 1

    return features, indices, spatial_shape, batch_size.to(device)
