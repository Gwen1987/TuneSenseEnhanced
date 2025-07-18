# ab_test_runner.py

from pymongo import MongoClient
import hashlib
from datetime import datetime
from dummy_users import DUMMY_USERS
from ab_config import AB_TESTS

# MongoDB setup
client = MongoClient("mongodb+srv://supertrooper:UofT1234@musiccluster.ix1va8y.mongodb.net/?retryWrites=true&w=majority&appName=musiccluster")
db = client["tunesense"]
ab_log_collection = db["ab_test_logs"]

def assign_variant(user_id: str, test_name: str) -> str:
    from ab_config import AB_TESTS

    test = AB_TESTS.get(test_name)
    if not test or not test.get("active"):
        return "A"  # fallback/default

    variants = test["variants"]
    key = f"{user_id}-{test_name}"
    hash_val = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    
    variant = variants[hash_val % len(variants)]
    print(f"🔍 TEST: {test_name}, USER: {user_id}, HASH: {hash_val}, MOD: {hash_val % len(variants)}, VARIANT: {variant}")
    return variant


def log_variant_assignment(user_id, test_name, variant):
    event = {
        "user_id": user_id,
        "test_name": test_name,
        "variant": variant,
        "timestamp": datetime.utcnow()
    }
    ab_log_collection.insert_one(event)

def simulate_ab_test():
    for user_id in DUMMY_USERS:
        variant = assign_variant(user_id, "recommendation_algorithm")
        log_variant_assignment(user_id, "recommendation_algorithm", variant)
        print(f"🧪 User: {user_id} -> Variant: {variant}")

if __name__ == "__main__":
    simulate_ab_test()
