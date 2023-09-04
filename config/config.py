def get_config():
    max_points_per_bubble = 100_000
    rings_per_bubble = 50
    assert max_points_per_bubble % rings_per_bubble == 0
    points_per_ring = int(max_points_per_bubble / rings_per_bubble)
    d_ring_embedding = 512
    point_features_div = 4
    assert d_ring_embedding % point_features_div == 0

    return\
        {
            "model_resolution": 0.5,
            "n_classes_model": 4,
            "d_ring_embedding": d_ring_embedding,
            "point_features_div": point_features_div,
            "batch_size": 1,
            "n_point_features": 3,  # x, y, z
            "max_points_per_bubble": max_points_per_bubble,
            "points_per_ring": points_per_ring,
            "rings_per_bubble": rings_per_bubble,
            "dropout": 0.0,
            "n_encoder_blocks": 6,
            "heads": 8,
            "learning_rate": 0.01
        }
