def get_config():
    max_points_per_bubble = 25_000
    rings_per_bubble = 10
    assert max_points_per_bubble % rings_per_bubble == 0
    points_per_ring = int(max_points_per_bubble / rings_per_bubble)

    return\
        {
            "model_resolution": 0.08,
            "max_points_per_bubble": max_points_per_bubble,
            "rings_per_bubble": rings_per_bubble,
            "points_per_ring": points_per_ring
        }
