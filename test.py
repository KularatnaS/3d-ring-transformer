from torch.utils.data import DataLoader
from dataset.dataset import TokenizedBubbleDataset, collate_fn, TrainingBubblesCreator
from config.config import get_config

from dataset.datautils import bubble_to_laz_file
from model.model import RingEmbedding

import logging
logging.basicConfig(level=logging.INFO)


bubble_to_laz_file('data/train-bubbles/pc_131_0_out_179.pt', 'bubble-vis.laz')

cf = get_config()
# BT = TrainingBubblesCreator(cf["max_points_per_bubble"], cf["points_per_ring"], 0.08)
# BT.run("data/train-data-mini", "data/train-bubbles", 15.0)

# dataset = TokenizedBubbleDataset('data/train-bubbles/', n_classes_model=4)
# dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=collate_fn)
#
# # get next batch
# iterator = iter(dataloader)
# point_tokens, label_tokens = next(iterator)
# x = point_tokens[0]
#
# print(x.shape)
# ring_embedding = RingEmbedding(cf["d_ring_embedding"], cf["n_point_features"])
# y = ring_embedding(x)
# print(y.shape)
# print(y)

# dataset = TokenizedBubbleDataset('train-bubbles/', n_classes_model=4)
#
# dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn)
# #
# # get next batch
# iterator = iter(dataloader)
# point_tokens, label_tokens = next(iterator)
# print(len(point_tokens))
# print(len(point_tokens[0]))
# print(point_tokens[0][0].shape)

from einops import rearrange
import torch
import torch.nn as nn
import numpy as np

# a = torch.rand(5, 3)
# a = rearrange(a, 'b c -> 1 c b')
# print(a.shape)
# conv1 = torch.nn.Conv1d(3, 10, 1)
# forward = conv1(a)
# print(forward.shape)
# print(forward)

from torch.autograd import Variable
# a = torch.tensor([[1, 1,  1],
#                   [1, 0, -1],
#                   [0, 0,  0],
#                   [1, 0,  1]], dtype=torch.float32)
# norm = nn.LayerNorm(4)
# print(norm(a))


