# simulate_ab_assignments.py

from uuid import uuid4
from ab_test_runner import assign_variant
from ab_logger import log_ab_event

test_name = "recommendation_knn_vs_cosine"

for _ in range(10):
    user_id = str(uuid4())
    variant = assign_variant(user_id, test_name)
    log_ab_event(user_id, test_name, variant, "assignment")
    print(f"🧪 User: {user_id} -> Variant: {variant}")
