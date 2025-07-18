import gradio as gr
from joblib import load
from ytmusicapi import YTMusic
import yt_dlp
import sys
import os
import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
import random 
# === PATH SETUP ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.downloader_core import extract_value
from extract_base import extract_features
from extract_extra import extract_extra_features

scaler = None
scaled_features = None
feature_cols = [
    "mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3",
    "chroma_0", "chroma_1", "chroma_2", "chroma_3",
    "tempo", "centroid", "rms", "zcr"
]
# === MongoDB Setup ===
client = MongoClient("mongodb+srv://supertrooper:UofT1234@musiccluster.ix1va8y.mongodb.net/?retryWrites=true&w=majority&appName=musiccluster")
db = client['tunesense']
meta_collection = db['tracks']
log_collection = db['ab_logs']


# === FFmpeg Path ===
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

# === Feature Cleaning Helper ===
def clean_feature(val):
    if isinstance(val, str) and val.startswith("["):
        try:
            return float(ast.literal_eval(val)[0])
        except Exception:
            return np.nan
    elif isinstance(val, (list, np.ndarray)):
        return float(val[0])
    elif isinstance(val, (float, int)):
        return float(val)
    else:
        return np.nan

# === Load Flattened Features and Fit Nearest Neighbors Model ===

try:
    feature_db = pd.read_csv("../outputs/flattened_features_with_extras.csv").dropna()

    def convert_cell(x):
        try:
            return eval(x)[0] if isinstance(x, str) and "[" in x else float(x)
        except Exception:
            return np.nan

    feature_cols = [col for col in feature_db.columns if col.startswith(('mfcc_', 'chroma_', 'spec_contrast_', 'tempo', 'centroid', 'rms', 'zcr'))]

    feature_db[feature_cols] = feature_db[feature_cols].applymap(convert_cell)
    feature_db = feature_db.dropna()
    feature_db[feature_cols] = feature_db[feature_cols].applymap(convert_cell)
    feature_db = feature_db.dropna()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_db[feature_cols].values)

except Exception as e:
    print(f"❌ Failed to load features: {e}")
    feature_db = None
    scaler = None
    scaled_features = None


# === Config ===
MODEL_PATH = "../widgets/tunesense_knn_model_FIXED.joblib"
LOG_PATH = "../logs/tunesense_logs.csv"

# === Load KNN Model and YTMusic Instance ===
try:
    knn_model = load(MODEL_PATH)
    print(f"✅ Loaded KNN model from {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Could not load model: {e}")
    knn_model = None
ytmusic = YTMusic()

# === Search YouTube Music ===
def search_ytmusic(query: str, limit: int = 5):
    results = ytmusic.search(query, filter="songs", limit=limit)
    output = []

    for item in results[:limit]:
        title = item.get("title")
        video_id = item.get("videoId")
        artist = item.get("artists", [{}])[0].get("name", "")
        thumbnails = item.get("thumbnails", [])
        thumbnail_url = thumbnails[-1]["url"] if thumbnails else None
        duration = item.get("duration")

        if title and video_id:
            display = f"{title} - {artist} {video_id}"
            output.append({
                "title": title,
                "artist": artist,
                "video_id": video_id,
                "thumbnail": thumbnail_url,
                "duration": duration
            })
            print(f"✅ Added result: {title} | video_id={video_id}")

    return output

# === Download Audio from YouTube ===
def download_audio(video_id: str, output_path="song_downloads"):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(id)s.%(ext)s',
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]
    }

    try:
        print(f"🔍 Downloading: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url)
            video_id = info.get("id", video_id)
            filename = f"{output_path}/{video_id}.mp3"
            print(f"(download_audio)✅ Returning file: {filename}")
            return f"✅ Downloaded: {info['title']}", filename
    except Exception as e:
        print(f"❌ Download error: {e}")
        return "❌ Download failed", None
    if features:
        recommendations = get_recommendations(features, variant=variant)
        log_ab_event(
            user_id,
            "recommendation_knn_vs_cosine",
            variant,
            "recommendation_served",
            {"video_id": video_id}
        )
    else:
        recommendations = ["❌ Feature extraction failed."]
# === Extract Base Audio Features ===
from time import time

def get_audio_features(video_id: str, download_dir="song_downloads") -> dict:
    file_path = os.path.join(download_dir, f"{video_id}.mp3")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    base = extract_features(file_path)
    extra = extract_extra_features(file_path)

    if not base or not extra:
        return None

    return {**base, **extra}




def get_recommendations(features: dict, variant: str = "A"):
    if not features or knn_model is None:
        return ["❌ Missing model or features."]

    try:
        flat_features = [extract_value(col, features) for col in feature_cols]
        flat_scaled = scaler.transform([flat_features])

        if variant == "A":
            # Use KNN model
            indices = knn_model.kneighbors(flat_scaled, return_distance=False)[0][:3]
        elif variant == "B":
            # Use Cosine similarity
            distances = cosine_similarity(flat_scaled, scaled_features)
            indices = distances.argsort()[0][-3:][::-1]
        else:
            return ["❌ Invalid variant."]

        recs = feature_db.iloc[indices]

        recommendations = []
        for i, (_, row) in enumerate(recs.iterrows()):
            filename = row.get("filename", "").replace(".mp3", "")
            meta = meta_collection.find_one({"filename": f"{filename}.mp3"}) or {}

            title = meta.get("title", "Unknown Title")
            artist = meta.get("artist_name", "Unknown Artist")
            duration = meta.get("duration", "")

            line = f"🎵 {i+1}. {title} – {artist}"
            if duration:
                line += f" ({duration})"
            recommendations.append(line)

        return recommendations

    except Exception as e:
        print(f"❌ Error in get_recommendations(): {e}")
        return ["❌ Recommendation error"]

# === Store Metadata in Mongo ===
def store_metadata(song_info):
    try:
        meta_collection.update_one(
            {"video_id": song_info["video_id"]},
            {"$set": {
                "title": song_info.get("title", ""),
                "artist": song_info.get("artist_name", ""),
                "duration": song_info.get("duration", ""),
                "thumbnail": song_info.get("thumbnail", "")
            }},
            upsert=True
        )
    except Exception as e:
        print(f"❌ Failed to store metadata: {e}")

# === AB TEST CONFIGURATION ===

AB_TESTS = {
    "top3_layout_test": {
        "description": "Test layout of Top 3 recommendations (list vs. card)",
        "variants": ["A", "B"],
        "primary_metric": "selected_song",
        "active": True
    }
}
import hashlib
from datetime import datetime

def assign_variant(user_id: str, test_name: str = "recommendation_knn_vs_cosine") -> str:
    """
    Deterministically assigns user to variant A or B using hash.
    Alternatively, you can assign randomly here.
    """
    variant = random.choice(["A", "B"])  # or your hashing logic
    log_ab_event(user_id, test_name, variant, "assignment")
    return variant

# MongoDB log collection

def log_ab_event(user_id: str, test_name: str, variant: str, event_type: str, metadata: dict = None):
    """Logs an AB test event to MongoDB."""
    try:
        log_entry = {
            "user_id": user_id,
            "test_name": test_name,
            "variant": variant,
            "event_type": event_type,  # e.g., "assigned", "recommendation_clicked"
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }
        log_collection.insert_one(log_entry)
        print(f"📬 Logged AB event: {log_entry}")
    except Exception as e:
        print(f"❌ Failed to log AB event: {e}")
