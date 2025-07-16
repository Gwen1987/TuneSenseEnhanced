'''from pathlib import Path
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import ast
import re


# Use pathlib for all paths
AUDIO_DIR = Path("data") / "fma small" / "fma_small_root" / "fma_small"
OUTPUT_FILE = Path("outputs/merged_features.csv")
LIMIT = 50  # Limit for processing files, can be adjusted

def extract_features(filepath):
    try:
        y, sr = librosa.load(str(filepath), sr=22050, mono=True, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        return {
            "mfcc": mfcc.mean(axis=1).tolist(),
            "chroma": chroma.mean(axis=1).tolist(),
            "spec_contrast": spec_contrast.mean(axis=1).tolist(),
            "tempo": tempo
        }
    except Exception as e:
        print(f"❌ Error with {filepath.name}: {e}")
        return None

def process_audio_files(limit=LIMIT):
    all_features = []
    count = 0

    for root, _, files in os.walk(AUDIO_DIR):
        for file in files:
            if file.endswith(".mp3"):
                full_path = Path(root) / file
                features = extract_features(full_path)
                if features:
                    features["filename"] = file
                    # Extract track ID (e.g., "000123.mp3" → 123)
                    track_id = int(re.sub(r"\D", "", file))
                    features["track_id"] = track_id
                    all_features.append(features)
                    count += 1  
                if count >= limit:
                    break
        if count >= limit:
            break
    return pd.DataFrame(all_features)    
        

def expand_list_column(df, column, prefix):
    # Parse string to list
    expanded = df[column].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    # Expand to new columns
    return pd.DataFrame(expanded.tolist(), columns=[f"{prefix}_{i}" for i in range(len(expanded.iloc[0]))])

def flatten_features(df):
    mfcc_df = expand_list_column(df, "mfcc", "mfcc")
    chroma_df = expand_list_column(df, "chroma", "chroma")
    spec_df = expand_list_column(df, "spec_contrast", "spec_contrast")
    tempo_df = df["tempo"].apply(lambda x: ast.literal_eval(x)[0] if pd.notna(x) else None)

    final_df = pd.concat([mfcc_df, chroma_df, spec_df], axis=1)
    final_df["tempo"] = tempo_df

    # Carry over additional columns
    for col in ["title", "artist_name", "genre_top", "filename"]:
        if col in df.columns:
            final_df[col] = df[col]

    return final_df



if __name__ == "__main__":
    df = process_audio_files()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved features for {len(df)} tracks to {OUTPUT_FILE}")
   
    # Step 2: Flatten (handled below by expanding and saving features)

    # Expand each list column
    mfcc_df = expand_list_column(df, "mfcc", "mfcc")
    chroma_df = expand_list_column(df, "chroma", "chroma")
    spec_df = expand_list_column(df, "spec_contrast", "spec_contrast")
    tempo_df = df["tempo"].apply(lambda x: x if not isinstance(x, str) else ast.literal_eval(x)[0] if pd.notna(x) else None)

    # Combine all
    final_df = pd.concat([mfcc_df, chroma_df, spec_df], axis=1)
    final_df["tempo"] = tempo_df

    for col in ["title", "artist_name", "genre_top", "filename"]:
        if col in df.columns:
            final_df[col] = df[col]

    # Save as flattened CSV
    flattened_file = Path("outputs/flattened_features.csv")
    flattened_file.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(flattened_file, index=False)
    print("✅ Saved flattened CSV with expanded features.")
    '''

from pathlib import Path
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import ast
import re

AUDIO_DIR = Path("data") / "fma small" / "fma_small_root" / "fma_small"
FEATURES_FILE = Path("outputs/merged_features.csv")
FLATTENED_FILE = Path("outputs/flattened_features.csv")
LIMIT = 50

def extract_features(filepath):
    try:
        y, sr = librosa.load(str(filepath), sr=22050, mono=True, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return {
            "mfcc": mfcc.mean(axis=1).tolist(),
            "chroma": chroma.mean(axis=1).tolist(),
            "spec_contrast": spec_contrast.mean(axis=1).tolist(),
            "tempo": [tempo]  # wrap in list for consistency
        }
    except Exception as e:
        print(f"❌ Error with {filepath}: {e}")
        return None

def process_audio_files(limit=LIMIT):
    all_features = []
    count = 0

    for root, _, files in os.walk(AUDIO_DIR):
        for file in files:
            if file.endswith(".mp3"):
                full_path = Path(root) / file
                features = extract_features(full_path)
                if features:
                    features["filename"] = file
                    features["track_id"] = int(file.replace(".mp3", ""))
                    all_features.append(features)
                    count += 1
                if count >= limit:
                    break
        if count >= limit:
            break

    return pd.DataFrame(all_features)

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

def flatten_features(df):
    mfcc_df = expand_list_column(df, "mfcc", "mfcc")
    chroma_df = expand_list_column(df, "chroma", "chroma")
    spec_df = expand_list_column(df, "spec_contrast", "spec_contrast")

    final_df = pd.concat([mfcc_df, chroma_df, spec_df], axis=1)
    final_df["tempo"] = tempo_df

    # Carry over additional columns
    for col in ["title", "artist_name", "genre_top", "filename", "track_id"]:
        if col in df.columns:
            final_df[col] = df[col]

    print(final_df.dtypes)
    return final_df


def parse_tempo(x):
    if isinstance(x, str):
        return ast.literal_eval(x)[0]
    elif isinstance(x, (list, np.ndarray)):
        return x[0]
    elif isinstance(x, (float, int)):  # fallback if tempo is scalar
        return x
    else:
        return None

if __name__ == "__main__":
    FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract features
    df = process_audio_files()
    df.to_csv(FEATURES_FILE, index=False, header=True)
    print(f"✅ Saved features for {len(df)} tracks to {FEATURES_FILE}")

    # Step 2: Flatten
    tempo_df = df["tempo"].apply(parse_tempo)
    flattened = flatten_features(df)
    flattened.to_csv(FLATTENED_FILE, index=False)
    print(f"✅ Saved flattened CSV to {FLATTENED_FILE}")

    
