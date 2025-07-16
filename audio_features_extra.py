import librosa

def extract_extra_features(filepath):
    try:
        y, sr = librosa.load(filepath, sr=22050, mono=True, duration=30)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y=y).mean()
        return {
            "centroid": centroid,
            "rms": rms,
            "zcr": zcr
        }
    except Exception as e:
        print(f"❌ Error extracting extras from {filepath}: {e}")
        return {"centroid": None, "rms": None, "zcr": None}
