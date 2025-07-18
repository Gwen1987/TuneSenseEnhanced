# dummy_users.py
import uuid

def generate_dummy_users(n=50):
    return [str(uuid.uuid4()) for _ in range(n)]

# For direct use
DUMMY_USERS = generate_dummy_users()
