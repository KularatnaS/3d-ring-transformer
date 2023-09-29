from pathlib import Path


def get_config():
    max_points_per_bubble = 100_000
    rings_per_bubble = 5
    assert max_points_per_bubble % rings_per_bubble == 0
    points_per_ring = int(max_points_per_bubble / rings_per_bubble)
    d_ring_embedding = 128
    n_extracted_point_features = 16

    return\
        {
            "model_resolution": 0.5,
            "n_classes_model": 4,
            "d_ring_embedding": d_ring_embedding,
            "n_extracted_point_features": n_extracted_point_features,
            "n_point_features": 3,  # x, y, z
            "max_points_per_bubble": max_points_per_bubble,
            "points_per_ring": points_per_ring,
            "rings_per_bubble": rings_per_bubble,
            "dropout": 0.0,
            "n_encoder_blocks": 6,
            "heads": 8,
            "ignore_index": -100,
            "ring_padding": 0.2,

            # Train config
            "laz_data_dir": "data/laz-data/",
            "bubble_data_dir": "data/bubbles/",
            "test_data_vis_dir": "data/test-data-vis/",
            "batch_size": 1,
            "learning_rate": 0.001,
            "model_folder": "train-artefacts/model-checkpoints/",
            "num_epochs": 100000,
            "preload": None,  # 'train-artefacts/model-checkpoints/checkpoint-29.pt',
            "tensorboard_log_dir": "train-artefacts/tensorboard-logs/"
        }


def get_weights_file_path(model_folder, epoch: str):
    model_basename = 'checkpoint-'
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('') / model_folder / model_filename)
