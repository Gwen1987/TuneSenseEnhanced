import gradio as gr
from joblib import load  # Import the load function from joblib
from ytmusicapi import YTMusic
import yt_dlp  # Import yt_dlp for downloading audio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from extract_features import extract_features
from audio_features_extra import extract_extra_features


os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"




# ===CONFIGURATION===
MODEL_PATH = "models/tunesense_knn_model.joblib"
LOG_PATH = "logs/tunesense_logs.csv" #May need to be updated to MongoDB later

# === LOAD MODEL ===
model = load(MODEL_PATH)  # Use joblib's load function
ytmusic = YTMusic()  # Create an instance of YTMusic


def search_ytmusic(query: str, limit: int = 5):
    # Function to search YouTube Music
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
                "title": display,
                "artist": artist,
                "video_id": video_id,
                "thumbnail": thumbnail_url,
                "duration": duration
            })
            print(f"✅ Added result: {title} | video_id={video_id}")

    return output  # Return a list of tuples (display_text, video_id)
    

def download_audio(video_id: str, output_path="song_downloads"):
    # Function to download audio from YouTube
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{output_path}/%(id)s.%(ext)s',
            'quiet': True,
            'postprocessors': [
                {
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }
            ]
        }
    try:
        print(f"🔍 Downloading: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url)
            video_id = info.get("id", video_id)  # fallback to the input if not found
            filename = f"{output_path}/{video_id}.mp3"
            print(f"(downloader_core.download_audio)🎯 Used ID for filename: {video_id}")

            print(f"(downloader_core.download_audio)✅ Returning file: {filename}")

            return f"✅ (downloader_core.download_audio)Downloaded: {info['title']}", filename
    except Exception as e:
        print(f"❌ (downloader_core.download_audio) Download error: {e}")
        return "❌ (downloader_core.download_audio)Download failed", None  # ✅ Return tuple with None
    

def get_audio_features(video_id: str, download_dir="song_downloads") -> dict:
    filename = f"{video_id}.mp3"
    file_path = os.path.join(download_dir, filename)

    if not os.path.exists(file_path):
        print(f"❌(downloader_core.get_audio_features) File not found: {file_path}")
        return None

    features = extract_features(file_path)
    return features

def get_recommendations(filepath):
    features = extract_features(filepath)  # existing base features
    if not features:
        return ["❌ (downloader_core.get_recommendations) Could not extract base features."]
    
    # 🔄 Add new features from audio_feature_extras
    extra_features = extract_extra_features(filepath)
    if not extra_features:
        return ["❌ (downloader_core.get_recommendations.extra_feature) Could not extract extra features."]

    features.update(extra_features)

    # 🚀 Prepare final feature vector for model (flatten into 1D list)
    flat_features = []
    for k in ["mfcc", "chroma", "spec_contrast", "tempo", "centroid", "rms", "zcr"]:
        v = features.get(k)
        if isinstance(v, list):
            flat_features.extend(v)
        elif isinstance(v, (float, int)):
            flat_features.append(v)

    # 🔮 Use model to find recommendations
    neighbors = model.kneighbors([flat_features], n_neighbors=3, return_distance=False)
    results = [f"🎵 Recommended Track #{i+1}" for i in neighbors[0]]
    return results


def get_radar_plot(video_id: str):
   # Placeholder plot logic
    import matplotlib.pyplot as plt
    import numpy as np

    features = ["Danceability", "Energy", "Valence", "Tempo", "Acousticness"]
    values = np.random.rand(len(features))
    values = np.append(values, values[0])
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, values, marker='o')
    ax.fill(angles, values, alpha=0.3)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_yticklabels([])
    ax.set_title("Audio Feature Radar")

    return fig