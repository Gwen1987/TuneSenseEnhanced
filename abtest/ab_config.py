# abtest/ab_config.py

AB_TESTS = {
    "recommendation_knn_vs_cosine": {
        "variants": ["A", "B"],  # A = KNN, B = Cosine
        "primary_metric": "engagement_rate",
        "secondary_metrics": ["click_through", "playlist_add"],
        "target_sample_size": 1000,
        "minimum_detectable_effect": 0.05,
        "statistical_power": 0.8,
        "significance_level": 0.05,
        "traffic_allocation": 1.0,  # full allocation
        "status": "running",
        "active": True,
        "description": "Testing effectiveness of KNN vs Cosine similarity for song recommendations"
    }
}
