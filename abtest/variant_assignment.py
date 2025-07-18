# abtest/variant_assignment.py

import hashlib
from ab_config import AB_TESTS

def assign_variant(user_id: str, test_name: str) -> str:
    test = AB_TESTS.get(test_name)
    if not test or not test.get("active"):
        return "A"  # fallback/default

    variants = test["variants"]
    key = f"{user_id}-{test_name}"
    hash_val = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    variant = variants[hash_val % len(variants)]
    print(f"🧪 User: {user_id} -> Variant: {variant}")
    return variant
