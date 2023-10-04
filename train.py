import os
from einops import rearrange
import numpy as np
import math
from tqdm import tqdm

import torch.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import collate_fn, TokenizedBubbleDataset
from config.config import get_config, get_weights_file_path
from model.model import build_classification_model
from dataset.datautils import save_as_laz_file
from train_utils.train_utils import run_validation
from train_utils.bubble_augmenter import BatchAugmenter

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
n_extracted_point_features = config["n_extracted_point_features"]
dropout = config["dropout"]
n_encoder_blocks = config["n_encoder_blocks"]
heads = config["heads"]
learning_rate = config["learning_rate"]
ignore_index = config["ignore_index"]
ring_padding = config["ring_padding"]
model_folder = config["model_folder"]
num_epochs = config["num_epochs"]
preload = config["preload"]
tensorboard_log_dir = config["tensorboard_log_dir"]
bubble_data_dir = config["bubble_data_dir"]
test_data_vis_dir = config["test_data_vis_dir"]
augment_train_data = config["augment_train_data"]

dirs_to_create = [model_folder, tensorboard_log_dir, test_data_vis_dir]
for dir_to_create in dirs_to_create:
    if not os.path.exists(dir_to_create):
        os.makedirs(dir_to_create)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using device:{device}")

# prepare train dataset
train_data_dir = os.path.join(bubble_data_dir, 'train-bubbles')
train_dataset = TokenizedBubbleDataset(data_dir=train_data_dir, n_classes_model=n_classes_model,
                                       rings_per_bubble=rings_per_bubble)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                              collate_fn=collate_fn)

# create model
model = build_classification_model(d_ring_embedding=d_ring_embedding, n_point_features=n_point_features,
                                   n_extracted_point_features=n_extracted_point_features, rings_per_bubble=rings_per_bubble,
                                   dropout=dropout, n_encoder_blocks=n_encoder_blocks, heads=heads,
                                   n_classes_model=n_classes_model, model_resolution=model_resolution).to(device)

# prepare validation dataset
val_data_dir = os.path.join(bubble_data_dir, 'val-bubbles')
val_dataset = TokenizedBubbleDataset(data_dir=val_data_dir, n_classes_model=n_classes_model,
                                     rings_per_bubble=rings_per_bubble)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                            collate_fn=collate_fn)

# prepare test dataset
test_data_dir = os.path.join(bubble_data_dir, 'test-bubbles')
test_dataset = TokenizedBubbleDataset(data_dir=test_data_dir, n_classes_model=n_classes_model,
                                        rings_per_bubble=rings_per_bubble)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                             collate_fn=collate_fn)

# define criterion and optimizer
criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Tensorboard
writer = SummaryWriter(tensorboard_log_dir)

initial_epoch = 0
if preload:
    LOGGER.info(f"Pre loading weights from {preload}")
    state = torch.load(preload)
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    model.load_state_dict(state['model_state_dict'])

# training loop
total_samples = len(train_dataset)
n_iterations = math.ceil(total_samples / batch_size)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
LOGGER.info(f'batch size: {batch_size}')
LOGGER.info(f'n_iterations: {n_iterations}')

for epoch in range(initial_epoch, num_epochs):
    train_loss_accumulation = 0.0
    batch_iterator = tqdm(train_dataloader, desc=f'Processing epoch {epoch}')
    for batch in batch_iterator:
        model.train()

        # data batch
        data = batch[0]
        if augment_train_data:
            batch_augmenter = BatchAugmenter(data, model_resolution)
            data = batch_augmenter.augment()
        data = data.to(device)

        labels = batch[1].type(torch.LongTensor).to(device)
        mask = batch[2].to(device)

        # forward pass
        y_predicted = model(data, mask)
        y_predicted = rearrange(y_predicted, 'a b c d -> (a b c) d')

        # calculate loss
        labels = rearrange(labels, 'a b c -> (a b c)')
        loss = criterion(y_predicted, labels)

        batch_iterator.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]["lr"]})

        train_loss_accumulation += loss.item()

        # backward pass
        loss.backward()

        # update weights / zero gradients
        optimizer.step()
        optimizer.zero_grad()

    train_loss_avg = train_loss_accumulation / n_iterations
    scheduler.step(train_loss_avg)

    # log the loss
    writer.add_scalar('train loss', train_loss_avg, epoch)
    writer.flush()

    # Save the model after each epoch
    model_filename = get_weights_file_path(model_folder=model_folder, epoch=f'{epoch}')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_filename)

    LOGGER.info(f"Running validation for epoch {epoch}")
    run_validation(model, device, val_dataloader, criterion, val_dataset, batch_size, writer, epoch)
