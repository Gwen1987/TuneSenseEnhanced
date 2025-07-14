from PyQt6.QtCore import Qt, QThreadPool, QSize, QObject, pyqtSignal, QRunnable
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QPushButton, QListWidget, QListWidgetItem, QMessageBox, QLabel
import requests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pymongo import MongoClient
import numpy as np
import librosa
import tempfile
import os
import yt_dlp
import joblib

# --- MongoDB Setup ---
client = MongoClient("mongodb+srv://supertrooper:UofT1234@musiccluster.ix1va8y.mongodb.net/?retryWrites=true&w=majority&appName=musiccluster")
db = client['tunesense']
collection = db['tracks']

# --- Load KNN model ---
knn_model = joblib.load("enhanced_widgets/tunesense_knn_model_FIXED.joblib")  # Adjusted path
features_columns = [
    "mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3",
    "chroma_0", "chroma_1", "tempo", "centroid", "rms", "zcr"
]


class ThumbnailSignalEmitter(QObject):
    signal = pyqtSignal(object, object)  # item, icon


class ThumbnailWorker(QRunnable):
    def __init__(self, url, item, signal_emitter):
        super().__init__()
        self.url = url
        self.item = item
        self.signal_emitter = signal_emitter

    def run(self):
        try:
            response = requests.get(self.url, timeout=5)
            response.raise_for_status()
            pixmap = QPixmap()
            pixmap.loadFromData(response.content)

            if not pixmap.isNull():
                icon = QIcon(pixmap.scaled(160, 90, Qt.AspectRatioMode.KeepAspectRatio))
                self.signal_emitter.signal.emit(self.item, icon)
        except Exception as e:
            print(f"❌ Thumbnail fetch error: {e}")


class RadarChartCanvas(FigureCanvas):
    def __init__(self, feature_dict, parent=None):
        fig, self.ax = plt.subplots(subplot_kw=dict(polar=True))
        super().__init__(fig)
        self.setParent(parent)
        self.plot(feature_dict)

    def plot(self, feature_dict):
        labels = list(feature_dict.keys())
        values = list(feature_dict.values())
        values += values[:1]  # Close the loop
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        self.ax.clear()
        self.ax.plot(angles, values, marker='o')
        self.ax.fill(angles, values, alpha=0.25)
        self.ax.set_xticks(angles[:-1])
        self.ax.set_xticklabels(labels)
        self.ax.set_title("Song Feature Radar Chart")


# --- Helper Functions ---
def get_audio_features(video_id, title):
    try:
        print(f"🎧 Extracting audio features for: {title}")

        with tempfile.TemporaryDirectory() as tmpdir:
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(tmpdir, 'temp.%(ext)s'),
                'quiet': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'noplaylist': True,
                'default_search': 'ytsearch1:' + title
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([title])

            print(f"📁 Files in tmpdir: {os.listdir(tmpdir)}")
            audio_path = [f for f in os.listdir(tmpdir) if f.endswith('.wav')][0]
            full_path = os.path.join(tmpdir, audio_path)
            y, sr = librosa.load(full_path)

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=4)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            tempo = librosa.beat.tempo(y=y, sr=sr)[0]
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            rms = np.mean(librosa.feature.rms(y=y)[0])
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=y)[0])

            return {
                'mfcc_0': np.mean(mfccs[0]),
                'mfcc_1': np.mean(mfccs[1]),
                'mfcc_2': np.mean(mfccs[2]),
                'mfcc_3': np.mean(mfccs[3]),
                'chroma_0': np.mean(chroma[0]),
                'chroma_1': np.mean(chroma[1]),
                'tempo': tempo,
                'centroid': centroid,
                'rms': rms,
                'zcr': zcr,
            }
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return None


def get_recommendations(feature_vector):
    try:
        print(f'[LOGGER] feature_vector = {feature_vector}')
        vector = [feature_vector[col] for col in features_columns]
        distances, indices = knn_model.kneighbors([vector], n_neighbors=4)
        rec_indices = indices[0][1:]  # skip self
        rec_vectors = knn_model._fit_X[rec_indices]
        return rec_vectors
    except Exception as e:
        print(f"❌ Recommendation error: {e}")
        return []


def fetch_metadata(vectors):
    print(f'[LOGGER]: vectors = {vectors}')
    track_ids = [int(vec[-1]) if isinstance(vec[-1], (int, float)) else vec[-1] for vec in vectors]
    return list(collection.find({"track_id": {"$in": track_ids}}))
