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
from pymongo import MongoClient
import random
from datetime import datetime

# === PATH SETUP ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from features.extract_base import extract_features
from features.extract_extra import extract_extra_features

# === MongoDB Setup ===
client = MongoClient("mongodb+srv://supertrooper:UofT1234@musiccluster.ix1va8y.mongodb.net/?retryWrites=true&w=majority&appName=musiccluster")
db = client['tunesense']
meta_collection = db['tracks']
log_collection = db['ab_logs']

# === FFmpeg Path ===
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

# === Load Flattened Features and Preprocess ===
try:
    feature_db = pd.read_csv("outputs/flattened_features_with_extras.csv")
    feature_db = feature_db.dropna()

    def convert_cell(x):
        if isinstance(x, str):
            try:
                val = ast.literal_eval(x)
                if isinstance(val, list):
                    return val[0]
                return float(val)
            except:
                return np.nan
        return x

    feature_cols = [col for col in feature_db.columns if col.startswith(('mfcc_', 'chroma_', 'spec_contrast_', 'tempo', 'centroid', 'rms', 'zcr'))]
    feature_db[feature_cols] = feature_db[feature_cols].apply(lambda col: col.map(convert_cell))
    feature_db = feature_db.dropna()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_db[feature_cols].values)

    # === Load KNN Models (cosine + euclidean) ===
    model_dir = os.path.dirname(__file__)
    knn_cosine = load(os.path.join(model_dir, "model_knn_cosine.pkl"))
    knn_euclidean = load(os.path.join(model_dir, "model_knn_euclidean.pkl"))

    print("✅ Loaded both KNN models")

except Exception as e:
    print(f"❌ Failed to initialize recommendation engine: {e}")
    feature_db = None
    knn_cosine = None
    knn_euclidean = None

# === Load YTMusic ===
ytusic = YTMusic()

# === Search YouTube Music ===
def search_ytmusic(query: str, limit: int = 5):
    results = ytusic.search(query, filter="songs", limit=limit)
    output = []

    for item in results[:limit]:
        title = item.get("title")
        video_id = item.get("videoId")
        artist = item.get("artists", [{}])[0].get("name", "")
        thumbnails = item.get("thumbnails", [])
        thumbnail_url = thumbnails[-1]["url"] if thumbnails else None
        duration = item.get("duration")

        if title and video_id:
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
            filename = f"{output_path}/{info['id']}.mp3"
            print(f"(download_audio)✅ Returning file: {filename}")
            return f"✅ Downloaded: {info['title']}", filename
    except Exception as e:
        print(f"❌ Download error: {e}")
        return "❌ Download failed", None

# === Extract Audio Features ===
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

# === AB Recommendation Engine ===
def extract_value(k, features_dict):
    base_key = k.split("_")[0]
    idx = k.split("_")[1] if "_" in k else None
    value = features_dict.get(base_key)

    if isinstance(value, list) and idx is not None:
        try:
            return float(value[int(idx)])
        except:
            return 0.0
    elif isinstance(value, np.ndarray) and idx is not None:
        try:
            return float(value[int(idx)])
        except:
            return 0.0
    elif isinstance(value, (float, int, np.generic)):
        return float(value)
    elif hasattr(value, '__iter__') and len(value) > 0 and isinstance(value[0], (float, int, np.generic, np.ndarray)):
        try:
            return float(value[0])
        except:
            return 0.0
    return 0.0

def get_recommendations(features: dict, variant: str = "A"):
    if not features or knn_cosine is None or knn_euclidean is None:
        return ["❌ Missing model or features."]

    try:
        flat_features = [extract_value(col, features) for col in feature_cols]
        flat_scaled = scaler.transform([flat_features])

        if variant == "A":
            indices = knn_cosine.kneighbors(flat_scaled, return_distance=False)[0]
            print("🔁 Variant A: cosine model used")
        elif variant == "B":
            indices = knn_euclidean.kneighbors(flat_scaled, return_distance=False)[0]
            print("🔁 Variant B: euclidean model used")
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

def assign_variant(user_id: str, test_name: str = "recommendation_knn_vs_cosine") -> str:
    variant = random.choice(["A", "B"])
    log_ab_event(user_id, test_name, variant, "assignment")
    return variant

def log_ab_event(user_id: str, test_name: str, variant: str, event_type: str, metadata: dict = None):
    try:
        log_entry = {
            "user_id": user_id,
            "test_name": test_name,
            "variant": variant,
            "event_type": event_type,
            "timestamp": datetime.utcnow(),
            "metadata": metadata or {}
        }
        log_collection.insert_one(log_entry)
        print(f"📬 Logged AB event: {log_entry}")
    except Exception as e:
        print(f"❌ Failed to log AB event: {e}")
