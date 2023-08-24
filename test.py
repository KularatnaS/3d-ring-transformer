import logging
logging.basicConfig(level=logging.INFO)
import config
from dataset.dataset import TrainingBubblesCreator
cf = config.get_config()
BT = TrainingBubblesCreator(cf["max_points_per_bubble"], cf["max_points_per_ring"], 0.08)
BT.run("data/train-data-mini", "data/train-bubbles", 15.0)

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

# from einops import rearrange