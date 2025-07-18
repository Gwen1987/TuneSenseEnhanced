# app/abtest/ab_logger.py

from pymongo import MongoClient
from datetime import datetime

# Setup MongoDB connection
client = MongoClient("mongodb+srv://supertrooper:UofT1234@musiccluster.ix1va8y.mongodb.net/?retryWrites=true&w=majority&appName=musiccluster")
db = client['tunesense']
log_collection = db['logs_abtest']

def log_ab_event(user_id: str, test_name: str, variant: str, event_type: str, metadata: dict = None):
    """
    Logs an AB test assignment or outcome event to MongoDB.

    Parameters:
    - user_id (str): The UUID of the user
    - test_name (str): Name of the AB test
    - variant (str): Assigned variant (e.g., "A", "B")
    - event_type (str): "assignment", "interaction", "click", "conversion", etc.
    - metadata (dict): Optional dict of additional fields (e.g., recommendations, timestamps)
    """
    log_entry = {
        "user_id": user_id,
        "test_name": test_name,
        "variant": variant,
        "event_type": event_type,
        "timestamp": datetime.utcnow(),
        "metadata": metadata or {}
    }
    
    try:
        log_collection.insert_one(log_entry)
        print(f"📦 Logged AB event: {log_entry}")
    except Exception as e:
        print(f"❌ Failed to log AB event: {e}")
