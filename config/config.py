from pathlib import Path
import datetime


def get_config():
    rings_per_bubble = 3
    points_per_ring = 150_000

    return\
        {
            "model_resolution": 0.5,
            "n_classes_model": 7,
            "d_ring_embedding": 128,
            "n_extracted_point_features": 16,
            "n_point_features": 3,  # x, y, z
            "points_per_ring": points_per_ring,
            "rings_per_bubble": rings_per_bubble,
            "max_points_per_bubble": points_per_ring * rings_per_bubble,
            "extra_rings_for_last_ring_padding": 2,
            "dropout": 0.01,
            "n_encoder_blocks": 3,
            "heads": 4,
            "ignore_index": -100,
            "ring_padding": 0.1,

            # Train config
            "min_rings_per_laz": rings_per_bubble + 1,
            "grid_resolution": 120.0,
            "laz_data_dir": "data/laz-data/",
            "bubble_data_dir": "data/bubbles/",
            "test_data_vis_dir": "data/test-data-vis/",

            "batch_size": 1,
            "learning_rate": 0.001,
            "model_folder": "train-artefacts/model-checkpoints/",
            "num_epochs": 100000,
            "preload": None,
            "tensorboard_log_dir": "train-artefacts/tensorboard-logs/" +
                                   datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
            "augment_train_data": True,

            # Test config
            "test_model_path": "train-artefacts/model-checkpoints/checkpoint-13.pt",
        }


def get_weights_file_path(model_folder, epoch: str):
    model_basename = 'checkpoint-'
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('') / model_folder / model_filename)
