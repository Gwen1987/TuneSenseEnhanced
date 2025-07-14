from PyQt6.QtCore import QThread, pyqtSignal
import pandas as pd
import joblib
import uuid
from datetime import datetime
from ab_testing_framework import ABTestManager, TuneSenseABTester, ExperimentConfig, MetricType, ExperimentStatus
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

class EnhancedRecommendationWorker(QThread):
    recommendations_ready = pyqtSignal(list)

    def __init__(self, input_row, user_id=None):
        super().__init__()
        self.input_row = input_row
        self.user_id = user_id or f"user_{uuid.uuid4().hex[:8]}"
        mongo_uri = "mongodb+srv://supertrooper:UofT1234@musiccluster.ix1va8y.mongodb.net/?retryWrites=true&w=majority&appName=musiccluster"
        self.ab_manager = ABTestManager(mongo_uri)
        self.ab_tester = TuneSenseABTester(self.ab_manager)
        self.ab_tester.set_user(self.user_id)

    def run(self):
        try:
            bundle = joblib.load('enhanced_widgets/tunesense_knn_model_FIXED.joblib')
            model = bundle["model"]
            feature_names = bundle["feature_names"]
            metadata = bundle["metadata"]

            row = self.input_row.copy()
            for feat in feature_names:
                val = row.get(feat, 0)
                if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
                    try:
                        row[feat] = float(val.strip("[]"))
                    except ValueError:
                        row[feat] = 0.0

            input_values = [row.get(f, 0) for f in feature_names]
            X_input = pd.DataFrame([input_values], columns=feature_names).astype(float)

            variant = self.ab_tester.get_variant("recommendation_algorithm")

            if variant == "treatment":
                distances, indices = model.kneighbors(X_input)
                similarities = 1 - distances[0]
                filtered = [(idx, sim) for idx, sim in zip(indices[0], similarities) if sim > 0.8]
                recommended_rows = metadata.iloc[[i for i, _ in filtered]].to_dict("records")
                avg_similarity = np.mean([sim for _, sim in filtered]) if filtered else 0.0
                print("[LOGGER] Treatment recommendations used")
            else:
                distances, indices = model.kneighbors(X_input)
                recommended_rows = metadata.iloc[indices[0]].to_dict("records")
                similarities = 1 - distances[0]
                avg_similarity = np.mean(similarities)
                print("[LOGGER] Control recommendations used")

            self.ab_tester.track_metric(
                experiment_name="recommendation_algorithm",
                metric_type=MetricType.SIMILARITY_SCORE,
                value=avg_similarity,
                metadata={
                    "variant": variant,
                    "approach": "filtered" if variant == "treatment" else "standard",
                    "num_recommendations": len(recommended_rows),
                    "timestamp": datetime.now().isoformat()
                }
            )

            variant_config = {
                "variant": variant,
                "k": 5 if variant == "treatment" else 3,
                "weighted": variant == "treatment"
            }
            for rec in recommended_rows:
                rec["_ab_variant"] = variant_config

            self.recommendations_ready.emit(recommended_rows)

        except Exception as e:
            print(f"❌ EnhancedRecommendationWorker error: {e}")
            self.recommendations_ready.emit([])

from sklearn.metrics import mean_squared_error, r2_score

def print_model_metrics():
    bundle = joblib.load("enhanced_widgets/tunesense_knn_model_FIXED.joblib")
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    df = pd.read_csv("final_merged_with_extras.csv")

    if "tempo" in df.columns and df["tempo"].dtype == object:
        df["tempo"] = df["tempo"].apply(lambda x: float(x.strip("[]")) if isinstance(x, str) and x.startswith("[") else x)

    X = df[feature_names].copy().fillna(0)
    distances, indices = model.kneighbors(X)
    reconstructed = X.iloc[indices[:, 1:]].mean(axis=1)

    mse = mean_squared_error(X.mean(axis=1), reconstructed)
    r2 = r2_score(X.mean(axis=1), reconstructed)

    print("📊 KNN Reconstruction Metrics")
    print(f"🧠 Mean Squared Error (MSE): {mse:.4f}")
    print(f"🎯 R² Score: {r2:.4f}")
