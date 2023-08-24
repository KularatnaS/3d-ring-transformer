def get_config():
    max_points_per_bubble = 25_000
    max_rings_per_bubble = 50
    assert max_points_per_bubble % max_rings_per_bubble == 0
    max_points_per_ring = int(max_points_per_bubble / max_rings_per_bubble)

    return\
        {
            "model_resolution": 0.08,
            "n_classes_model": 4,
            "max_points_per_bubble": max_points_per_bubble,
            "max_points_per_ring": max_points_per_ring,
        }
