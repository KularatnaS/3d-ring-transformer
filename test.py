import os

import torch.nn
from torch.utils.data import DataLoader

from dataset.dataset import collate_fn, TokenizedBubbleDataset
from config.config import get_config
from model.model import build_classification_model

from train_utils.train_utils import run_test

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
test_model_path = config["test_model_path"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using device:{device}")

# create model
model = build_classification_model(d_ring_embedding=d_ring_embedding, n_point_features=n_point_features,
                                   n_extracted_point_features=n_extracted_point_features, rings_per_bubble=rings_per_bubble,
                                   dropout=dropout, n_encoder_blocks=n_encoder_blocks, heads=heads,
                                   n_classes_model=n_classes_model, model_resolution=model_resolution).to(device)

# prepare test dataset
test_data_dir = os.path.join(bubble_data_dir, 'test-bubbles')
test_dataset = TokenizedBubbleDataset(data_dir=test_data_dir, n_classes_model=n_classes_model,
                                        rings_per_bubble=rings_per_bubble)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                             collate_fn=collate_fn)

LOGGER.info(f"Pre loading weights from {test_model_path}")
state = torch.load(test_model_path)
model.load_state_dict(state['model_state_dict'])
epoch = state['epoch']

LOGGER.info(f"Running test for epoch {epoch}")
run_test(model, device, test_dataloader, test_data_vis_dir, n_classes_model, rings_per_bubble, points_per_ring,
         ring_padding)

