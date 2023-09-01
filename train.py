import torch.nn

from dataset.dataset import collate_fn, TokenizedBubbleDataset
from dataset.datautils import save_as_laz_file
from config.config import get_config
from model.model import build_classification_model

from torch.utils.data import DataLoader

from einops import rearrange
import numpy as np
import math

import logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# prepare config
config = get_config()

max_points_per_bubble = config["max_points_per_bubble"]
points_per_ring = config["points_per_ring"]
rings_per_bubble = config["rings_per_bubble"]
n_point_features = config["n_point_features"]
model_resolution = config["model_resolution"]
n_classes_model = config["n_classes_model"]
batch_size = config["batch_size"]
d_ring_embedding = config["d_ring_embedding"]
point_features_div = config["point_features_div"]
dropout = config["dropout"]
n_encoder_blocks = config["n_encoder_blocks"]
heads = config["heads"]
learning_rate = config["learning_rate"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# prepare train dataset
train_data_dir = 'data/train-bubbles/'
dataset = TokenizedBubbleDataset(data_dir=train_data_dir, n_classes_model=n_classes_model,
                                 rings_per_bubble=rings_per_bubble)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

# create model
model = build_classification_model(d_ring_embedding=d_ring_embedding, n_point_features=n_point_features,
                                   point_features_div=point_features_div, rings_per_bubble=rings_per_bubble,
                                   dropout=dropout, n_encoder_blocks=n_encoder_blocks, heads=heads,
                                   n_classes_model=n_classes_model).to(device)

# prepare validation dataset
val_data_dir = 'data/val-bubbles/'
val_dataset = TokenizedBubbleDataset(data_dir=val_data_dir, n_classes_model=n_classes_model,
                                        rings_per_bubble=rings_per_bubble)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=2,
                            collate_fn=collate_fn)
def run_validation(model, device, val_dataloader):
    model.eval()
    iterator = iter(val_dataloader)
    batch = next(iterator)

    point_tokens = batch[0].to(device)
    mask = batch[2].to(device)

    with torch.no_grad():
        y_predicted = model(point_tokens, mask)  # [batch, n_rings, n_points_per_ring, n_classes_model]
        y_predicted = rearrange(y_predicted, 'a b c d -> (a b c) d')
        print(y_predicted.shape)
        labels = torch.argmax(y_predicted, dim=1)
        # detach from gpu
        labels = labels.cpu().numpy()
        print(np.unique(labels))
        # save as laz file
        all_points = rearrange(point_tokens, '1 a b c -> (a b) c').cpu().numpy()
        save_as_laz_file(points=all_points, classification=labels, filename='data/visualise-val/view.laz')

# define criterion and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-9)

# training loop
num_epochs = 10000000
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / batch_size)

print('batch size:', batch_size)

for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):

        # data batch
        data = batch[0].to(device)
        labels = batch[1].to(device)
        mask = batch[2].to(device)

        # forward pass
        y_predicted = model(data, mask)
        y_predicted = rearrange(y_predicted, 'a b c d -> (a b c) d')

        # calculate loss
        labels = rearrange(labels, 'a b c d -> (a b c) d')
        labels = torch.argmax(labels, dim=1)  # one hot encoded to class labels
        loss = criterion(y_predicted, labels)

        # backward pass
        loss.backward()

        # update weights / zero gradients
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 5 == 0:
            LOGGER.info(f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_iterations}, train loss: {loss.item():.4f}')
        if (i + 1) % 100 == 0:
            run_validation(model, device, val_dataloader=val_dataloader)