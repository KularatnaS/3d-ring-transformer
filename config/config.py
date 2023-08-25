def get_config():
    max_points_per_bubble = 100_000
    rings_per_bubble = 50
    assert max_points_per_bubble % rings_per_bubble == 0
    points_per_ring = int(max_points_per_bubble / rings_per_bubble)

    return\
        {
            "model_resolution": 0.08,
            "n_classes_model": 4,
            "d_ring_embedding": 256,
            "batch_size": 2,
            "n_point_features": 3,  # x, y, z
            "max_points_per_bubble": max_points_per_bubble,
            "points_per_ring": points_per_ring,
        }
