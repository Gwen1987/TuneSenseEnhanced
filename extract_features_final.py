from pathlib import Path
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import ast

# Paths and config
AUDIO_DIR = Path("data") / "fma_large" / "fma_large_root" / "fma_large"
FEATURES_FILE = Path("outputs/merged_features.csv")
FLATTENED_FILE = Path("outputs/flattened_features.csv")
LIMIT = 5000
filepath = Path("song_downloads")

def extract_features(filepath):
    filepath = Path(filepath)  # ✅ Convert to Path object if passed as a string

    try:
        print(f"🔍 Processing: {filepath.name}")
        y, sr = librosa.load(str(filepath), sr=22050, mono=True, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return {
            "mfcc": mfcc.mean(axis=1).tolist(),
            "chroma": chroma.mean(axis=1).tolist(),
            "spec_contrast": spec_contrast.mean(axis=1).tolist(),
            "tempo": [tempo]
        }
    except Exception as e:
        print(f"(extract_features_final.extract_features)❌ Error with {filepath}: {e}")
        return None

def process_audio_files(limit=LIMIT):
    all_features = []
    count = 0
    audio_files = []

    # Collect all .mp3 paths
    for root, _, files in os.walk(AUDIO_DIR):
        for file in files:
            if file.endswith(".mp3"):
                audio_files.append(Path(root) / file)

    print(f"🎧 Found {len(audio_files)} audio files. Extracting up to {limit} tracks...")

    # Process with tqdm progress bar
    for file in tqdm(audio_files[:limit], desc="🚀 Extracting features"):
        features = extract_features(file)
        if features:
            features["filename"] = file.name
            features["track_id"] = int(file.stem)
            all_features.append(features)

    return pd.DataFrame(all_features)

def parse_tempo(x):
    if isinstance(x, str):
        return ast.literal_eval(x)[0]
    elif isinstance(x, (list, np.ndarray)):
        return x[0]
    elif isinstance(x, (float, int)):
        return x
    else:
        return None

def expand_list_column(df, column, prefix):
    def safe_parse(x):
        if isinstance(x, str):
            return ast.literal_eval(x)
        elif isinstance(x, (list, np.ndarray)):
            return x
        else:
            return []
    expanded = df[column].apply(safe_parse)
    return pd.DataFrame(expanded.tolist(), columns=[f"{prefix}_{i}" for i in range(len(expanded.iloc[0]))])

def flatten_features(df, tempo_df):
    mfcc_df = expand_list_column(df, "mfcc", "mfcc")
    chroma_df = expand_list_column(df, "chroma", "chroma")
    spec_df = expand_list_column(df, "spec_contrast", "spec_contrast")

    final_df = pd.concat([mfcc_df, chroma_df, spec_df], axis=1)
    final_df["tempo"] = tempo_df

    for col in ["title", "artist_name", "genre_top", "filename", "track_id"]:
        if col in df.columns:
            final_df[col] = df[col]

    return final_df

if __name__ == "__main__":
    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract features
    df = process_audio_files()
    df.to_csv(FEATURES_FILE, index=False, header=True)
    print(f"✅ Saved features for {len(df)} tracks to {FEATURES_FILE}")

    # Step 2: Flatten
    tempo_df = df["tempo"].apply(parse_tempo)
    flattened = flatten_features(df, tempo_df)
    flattened.to_csv(FLATTENED_FILE, index=False)
    print(f"✅ Saved flattened CSV to {FLATTENED_FILE}")
