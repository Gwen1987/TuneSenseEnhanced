import uuid
import random
import time
from abtest.ab_test_runner import assign_variant
from abtest.ab_logger import log_ab_event
from abtest.ab_config import AB_TESTS

TEST_NAME = "recommendation_knn_vs_cosine"

def simulate_user(user_id):
    variant = assign_variant(TEST_NAME, user_id)
    
    # Log assignment
    log_ab_event(user_id, TEST_NAME, variant, "assignment")

    # Simulate behavior
    interacted = random.random() < 0.6  # 60% engagement rate
    if interacted:
        metadata = {
            "song_id": str(uuid.uuid4()),
            "clicked": True,
            "timestamp": str(time.time()),
            "duration": random.randint(5, 120)  # seconds listened
        }
        log_ab_event(user_id, TEST_NAME, variant, "interaction", metadata)

def run_simulation(num_users=10000, delay=0.01):
    print(f"🚀 Starting simulation for {num_users} users...")
    for _ in range(num_users):
        user_id = str(uuid.uuid4())
        simulate_user(user_id)
        time.sleep(delay)  # simulate traffic spread

if __name__ == "__main__":
    run_simulation(num_users=10000, delay=0.01)  # Run overnight? Set delay to 0 or low
