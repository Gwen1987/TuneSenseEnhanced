import librosa

def extract_extra_features(filepath):
    try:
        print(f"🎧 [extract_extra_features] Processing: {filepath}")
        y, sr = librosa.load(filepath, sr=22050, mono=True, duration=30)

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y=y).mean()

        print(f"✅ [extract_extra_features] centroid={centroid:.2f}, rms={rms:.5f}, zcr={zcr:.5f}")
        
        return {
            "centroid": [centroid],
            "rms": [rms],
            "zcr": [zcr]
        }
    except Exception as e:
        print(f"❌ [extract_extra_features] Error from {filepath}: {e}")
        return {
            "centroid": [None],
            "rms": [None],
            "zcr": [None]
        }
