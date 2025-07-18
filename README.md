## 🎿 TuneSense

A content-based song recommendation engine using audio features extracted via Librosa and KNN similarity search. Includes:

* 🔍 Search + download via YouTube Music
* 🧠 Feature extraction (MFCCs, chroma, contrast, etc.)
* 🧪 A/B testing framework for evaluating recommendation strategies
* 📊 Audio feature visualization with radar plots
* 💃 MongoDB logging of user actions and variant performance

---

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/gwen1987/TuneSenseEnhanced.git
cd TuneSenseEnhanced
```

### 2. Create a virtual environment

```bash
python3.9 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Note: If `yt-dlp` fails due to `websockets` conflicts, comment it out and use mock audio instead.

---

### 4. Run the app

```bash
python app/app.py
```

---

## 🧪 A/B Testing

TuneSense includes a basic A/B framework that:

* Randomly assigns each user to Variant A or B
* Logs interactions to MongoDB Atlas
* Allows post-hoc analysis of variant performance

See `downloader_core.py` for logging structure and `mongodb_config.py` for your connection string.

---

## 🧠 Recommendation Logic

* Uses KNN trained on Librosa-extracted features
* Two variants supported:

  * `KNN-Euclidean`
  * `KNN-Cosine`

Models are saved as:

* `model_knn_euclidean.pkl`
* `model_knn_cosine.pkl`

---

## 📊 Feature Extraction

Features include:

* 13 MFCCs
* 12 Chroma features
* 7 Spectral contrast bands
* Tempo, Centroid, RMS, ZCR

Total: 36 audio features per track.

---

## 🖼️ Visualization

Each selected track includes a radar chart plot of its features for user-friendly interpretability.

---

## 🧪 MongoDB Logging

```python
log_event(user_id, variant, event_type, track_id, timestamp)
```

All logs are stored in the `ab_logs` collection in your Atlas cluster.

---

## 🙋‍♀️ Author

Gwen Seymour — [LinkedIn](https://www.linkedin.com/in/gwenseymour) | [GitHub](https://github.com/gwen1987)
