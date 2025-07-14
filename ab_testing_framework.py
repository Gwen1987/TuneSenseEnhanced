import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random

# Add these imports for AB testing
from datetime import datetime
import hashlib
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pymongo import MongoClient


class MetricType(Enum):
    """Types of metrics that can be tracked in AB tests"""
    SIMILARITY_SCORE = "similarity_score"
    CLICK_THROUGH_RATE = "click_through_rate"
    PLAY_RATE = "play_rate"
    SKIP_RATE = "skip_rate"
    LIKE_RATE = "like_rate"
    SESSION_DURATION = "session_duration"
    CONVERSION_RATE = "conversion_rate"
    RECOMMENDATION_DIVERSITY = "recommendation_diversity"
    USER_SATISFACTION = "user_satisfaction"
    ENGAGEMENT_SCORE = "engagement_score"

class ExperimentStatus(Enum):
    """Status of AB test experiments"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class ExperimentConfig:
    """Configuration for AB test experiments"""
    name: str
    description: str
    status: ExperimentStatus
    start_date: datetime
    end_date: Optional[datetime] = None
    
    # Variant configuration
    variants: Dict[str, float] = field(default_factory=lambda: {"control": 0.5, "treatment": 0.5})
    
    # Metrics to track
    primary_metric: MetricType = MetricType.SIMILARITY_SCORE
    secondary_metrics: List[MetricType] = field(default_factory=list)
    
    # Sample size and power analysis
    target_sample_size: int = 1000
    minimum_detectable_effect: float = 0.05
    statistical_power: float = 0.8
    significance_level: float = 0.05
    
    # Experiment parameters
    traffic_allocation: float = 1.0  # What percentage of users to include
    user_segments: List[str] = field(default_factory=list)  # User segments to include
    
    # Metadata
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate experiment configuration"""
        # Ensure variants sum to 1.0
        total_allocation = sum(self.variants.values())
        if abs(total_allocation - 1.0) > 0.001:
            raise ValueError(f"Variant allocations must sum to 1.0, got {total_allocation}")
        
        # Validate traffic allocation
        if not 0 < self.traffic_allocation <= 1.0:
            raise ValueError("Traffic allocation must be between 0 and 1")
        
        # Validate dates
        if self.end_date and self.end_date <= self.start_date:
            raise ValueError("End date must be after start date")
    
    def get_variant_for_user(self, user_id: str) -> str:
        """Determine which variant a user should be assigned to"""
        # Create deterministic hash
        hash_input = f"{user_id}_{self.name}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Normalize to 0-1 range
        normalized_hash = (hash_value % 10000) / 10000
        
        # Check if user is in experiment (traffic allocation)
        if normalized_hash > self.traffic_allocation:
            return "excluded"
        
        # Assign to variant based on allocation
        cumulative_allocation = 0
        for variant, allocation in self.variants.items():
            cumulative_allocation += allocation
            if normalized_hash <= cumulative_allocation:
                return variant
        
        # Fallback to control
        return "control"
    
    def is_active(self) -> bool:
        """Check if experiment is currently active"""
        if self.status != ExperimentStatus.ACTIVE:
            return False
        
        now = datetime.now()
        if now < self.start_date:
            return False
        
        if self.end_date and now > self.end_date:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "variants": self.variants,
            "primary_metric": self.primary_metric.value,
            "secondary_metrics": [m.value for m in self.secondary_metrics],
            "target_sample_size": self.target_sample_size,
            "minimum_detectable_effect": self.minimum_detectable_effect,
            "statistical_power": self.statistical_power,
            "significance_level": self.significance_level,
            "traffic_allocation": self.traffic_allocation,
            "user_segments": self.user_segments,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary"""
        return cls(
            name=data["name"],
            description=data["description"],
            status=ExperimentStatus(data["status"]),
            start_date=datetime.fromisoformat(data["start_date"]),
            end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            variants=data["variants"],
            primary_metric=MetricType(data["primary_metric"]),
            secondary_metrics=[MetricType(m) for m in data.get("secondary_metrics", [])],
            target_sample_size=data.get("target_sample_size", 1000),
            minimum_detectable_effect=data.get("minimum_detectable_effect", 0.05),
            statistical_power=data.get("statistical_power", 0.8),
            significance_level=data.get("significance_level", 0.05),
            traffic_allocation=data.get("traffic_allocation", 1.0),
            user_segments=data.get("user_segments", []),
            created_by=data.get("created_by", ""),
            created_at=datetime.fromisoformat(data["created_at"]),
            tags=data.get("tags", [])
        )

class ABTestManager:
    """Manages AB test configurations and user assignments"""
    
    def __init__(self, mongo_uri):
        self.mongo_uri = mongo_uri
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.client = MongoClient(mongo_uri)
        self.db = self.client.get_database("TuneSenseAB")  # or custom name
        self.events_collection = self.db.get_collection("ab_events")
        print(f"✅ Connected to MongoDB: {mongo_uri}")

        
    def create_experiment(self, config: ExperimentConfig):
        """Create a new experiment"""
        if config.name in self.experiments:
            raise ValueError(f"Experiment {config.name} already exists")
        
        self.experiments[config.name] = config
        print(f"Created experiment: {config.name}")
        # In real implementation, save to MongoDB
        
    def get_experiment(self, experiment_name: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration"""
        return self.experiments.get(experiment_name)
    
    def update_experiment_status(self, experiment_name: str, status: ExperimentStatus):
        """Update experiment status"""
        if experiment_name not in self.experiments:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        self.experiments[experiment_name].status = status
        print(f"Updated experiment {experiment_name} status to {status.value}")
    
    def get_active_experiments(self) -> List[ExperimentConfig]:
        """Get all active experiments"""
        return [exp for exp in self.experiments.values() if exp.is_active()]
        
    def get_user_variant(self, user_id: str, experiment_name: str) -> str:
        """Get variant assignment for a user"""
        experiment = self.get_experiment(experiment_name)
        if not experiment:
            raise ValueError(f"Experiment {experiment_name} not found")
        
        if not experiment.is_active():
            return "excluded"
        
        return experiment.get_variant_for_user(user_id)
    
    def log_event(self, user_id: str, experiment_name: str, event_type: str, 
                metric_type: MetricType, value: float, metadata: Dict[str, Any] = None):
        experiment = self.get_experiment(experiment_name)
        if not experiment:
            print(f"⚠️ Experiment {experiment_name} not found")
            return

        variant = self.get_user_variant(user_id, experiment_name)
        event = {
            "user_id": user_id,
            "experiment_name": experiment_name,
            "variant": variant,
            "event_type": event_type,
            "metric_type": metric_type.value,
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        try:
            self.events_collection.insert_one(event)
            print(f"📥 Event logged to MongoDB: {event_type} ({metric_type.value})")
        except Exception as e:
            print(f"❌ MongoDB log failed: {e}")


    def close(self):
        """Close database connections"""
        print("Closing AB test manager connections")

class TuneSenseABTester:
    """Handles AB testing for TuneSense music recommendations"""

    def __init__(self, ab_manager):
        self.ab_manager = ab_manager
        self.user_id = None
        self.active_experiments = {}

    def set_user(self, user_id):
        """Set the current user for testing"""
        self.user_id = user_id

    def get_variant(self, experiment_name):
        """Get user's variant for an experiment"""
        if not self.user_id:
            raise ValueError("User ID not set")

        if experiment_name not in self.active_experiments:
            variant = self.ab_manager.get_user_variant(self.user_id, experiment_name)
            self.active_experiments[experiment_name] = variant

        return self.active_experiments[experiment_name]

    def get_recommendation_config(self, experiment_name: str = "recommendation_algorithm_v1") -> dict:
        """Return config settings for the user's assigned recommendation variant"""
        variant = self.get_variant(experiment_name)
        config_by_variant = {
            "control": {"k": 3, "weighted": False},
            "treatment": {"k": 5, "weighted": True}
        }
        return config_by_variant.get(variant, config_by_variant["control"])

    def track_event(self, experiment_name: str, event_type: str, metadata: Dict[str, Any] = None):
        """Track an event for AB testing"""
        if not self.user_id:
            raise ValueError("User ID not set")

        self.ab_manager.log_event(
            user_id=self.user_id,
            experiment_name=experiment_name,
            event_type=event_type,
            metric_type=MetricType.ENGAGEMENT_SCORE,
            value=1.0,
            metadata=metadata or {}
        )

    def track_metric(self, experiment_name: str, metric_type: MetricType, value: float, metadata: Dict[str, Any] = None):
        """Track a metric for AB testing"""
        if not self.user_id:
            raise ValueError("User ID not set")

        self.ab_manager.log_event(
            user_id=self.user_id,
            experiment_name=experiment_name,
            event_type="metric_tracked",
            metric_type=metric_type,
            value=value,
            metadata=metadata or {}
        )

    def log_search_query(self, query: str):
        """Log a user search event"""
        if not self.user_id:
            raise ValueError("User ID not set")

        self.track_event(
            experiment_name="recommendation_algorithm_v1",
            event_type="search_query",
            metadata={"query": query}
        )

    def log_video_selection(self, video_data: dict):
        """Log a video selection event"""
        if not self.user_id:
            raise ValueError("User ID not set")

        self.track_event(
            experiment_name="recommendation_algorithm_v1",
            event_type="video_selected",
            metadata={
                "video_title": video_data.get("title", "unknown"),
                "video_id": video_data.get("id", "unknown"),
                "duration": video_data.get("duration", "unknown"),
                "channel": video_data.get("channel", "unknown")
            }
        )

    def log_recommendation_click(self, recommendation: dict):
        """Log a recommendation click"""
        if not self.user_id:
            raise ValueError("User ID not set")

        self.track_metric(
            experiment_name="recommendation_algorithm_v1",
            metric_type=MetricType.CLICK_THROUGH_RATE,
            value=1.0,
            metadata={
                "track_name": recommendation.get("track_name", "unknown"),
                "artist_name": recommendation.get("artist_name", "unknown"),
                "variant": recommendation.get("_ab_variant", {}).get("variant")
            }
        )
class MusicRecommendationABTester:
    def __init__(self, features_df, user_id=None):
        """
        Initialize with your CSV of music features and AB testing
        Expected columns: track_id, mfcc_1, mfcc_2, ..., chroma_1, chroma_2, ..., 
                         spectral_contrast, tempo, spectral_centroid, rms_energy, zcr
        """
        self.df = features_df
        self.scaler = StandardScaler()
        self.user_id = user_id
        
        # Initialize AB testing
        mongo_uri = "mongodb+srv://supertrooper:UofT1234@musiccluster.ix1va8y.mongodb.net/?retryWrites=true&w=majority&appName=musiccluster"
        self.ab_manager = ABTestManager(mongo_uri)
        self.ab_tester = TuneSenseABTester(self.ab_manager)
        
        # Create default experiment configuration
        self.setup_default_experiment()
        
        if self.user_id:
            self.ab_tester.set_user(self.user_id)
        
        self.prepare_features()
    
    def setup_default_experiment(self):
        """Setup default recommendation algorithm experiment"""
        default_config = ExperimentConfig(
            name="recommendation_algorithm",
            description="Test two-stage filtering vs standard recommendation",
            status=ExperimentStatus.ACTIVE,
            start_date=datetime.now(),
            variants={"control": 0.5, "treatment": 0.5},
            primary_metric=MetricType.SIMILARITY_SCORE,
            secondary_metrics=[
                MetricType.CLICK_THROUGH_RATE,
                MetricType.PLAY_RATE,
                MetricType.USER_SATISFACTION
            ],
            target_sample_size=1000,
            minimum_detectable_effect=0.05,
            created_by="system"
        )
        
        try:
            self.ab_manager.create_experiment(default_config)
        except ValueError:
            # Experiment already exists
            pass
        
    def set_user(self, user_id):
        """Set user for AB testing"""
        self.user_id = user_id
        self.ab_tester.set_user(user_id)
    
    def get_recommendation_variant(self, experiment_name="recommendation_algorithm"):
        """Get the recommendation variant for current user"""
        if not self.user_id:
            raise ValueError("User ID not set")
        return self.ab_tester.get_variant(experiment_name)
    
    def get_recommendations(self, query_track_idx, experiment_name="recommendation_algorithm"):
        """
        Get recommendations using AB testing to determine approach
        """
        if not self.user_id:
            # Default to standard recommendation if no user set
            return self.standard_recommendation(query_track_idx)
        
        # Get user's variant
        variant = self.get_recommendation_variant(experiment_name)
        
        # Track recommendation request
        self.ab_tester.track_event(
            experiment_name, 
            "recommendation_request",
            {"query_track_idx": query_track_idx}
        )
        
        # Apply appropriate recommendation approach based on variant
        if variant == "treatment":
            recommendations, similarities = self.two_stage_filtering(query_track_idx)
            approach = "two_stage_filtering"
        else:
            recommendations, similarities = self.standard_recommendation(query_track_idx)
            approach = "standard"
        
        # Track recommendation delivery with proper metrics
        self.ab_tester.track_metric(
            experiment_name,
            MetricType.SIMILARITY_SCORE,
            float(np.mean(similarities)),
            {
                "approach": approach,
                "num_recommendations": len(recommendations),
                "top_similarity": float(np.max(similarities)),
                "query_track_idx": query_track_idx
            }
        )
        
        return recommendations, similarities
    
    def track_user_interaction(self, experiment_name: str, interaction_type: str, 
                             track_idx: Optional[int] = None, metadata: Dict[str, Any] = None):
        """
        Track user interactions for AB testing analysis
        interaction_type: 'click', 'play', 'skip', 'like', 'dislike'
        """
        if not self.user_id:
            return
        
        # Map interaction types to metric types
        interaction_to_metric = {
            'click': MetricType.CLICK_THROUGH_RATE,
            'play': MetricType.PLAY_RATE,
            'skip': MetricType.SKIP_RATE,
            'like': MetricType.LIKE_RATE,
            'dislike': MetricType.LIKE_RATE,  # Negative value
            'session_end': MetricType.SESSION_DURATION
        }
        
        metric_type = interaction_to_metric.get(interaction_type, MetricType.ENGAGEMENT_SCORE)
        
        # Determine metric value
        if interaction_type == 'dislike':
            value = 0.0  # Negative engagement
        elif interaction_type == 'session_end' and metadata and 'duration' in metadata:
            value = float(metadata['duration'])
        else:
            value = 1.0  # Positive engagement
        
        event_metadata = {
            "interaction_type": interaction_type,
            "track_idx": track_idx,
            **(metadata or {})
        }
        
        self.ab_tester.track_metric(
            experiment_name,
            metric_type,
            value,
            event_metadata
        )
    
    def run_live_ab_test(self, user_ids, test_tracks, experiment_name="recommendation_algorithm"):
        """
        Run live AB test with actual users
        """
        results = {
            'user_id': [],
            'variant': [],
            'track_id': [],
            'approach': [],
            'avg_similarity': [],
            'top_similarity': [],
            'similarity_std': []
        }
        
        for user_id in user_ids:
            self.set_user(user_id)
            
            for track_idx in test_tracks:
                try:
                    # Get recommendations using AB testing
                    recommendations, similarities = self.get_recommendations(track_idx, experiment_name)
                    variant = self.get_recommendation_variant(experiment_name)
                    approach = "two_stage_filtering" if variant == "treatment" else "standard"
                    
                    results['user_id'].append(user_id)
                    results['variant'].append(variant)
                    results['track_id'].append(track_idx)
                    results['approach'].append(approach)
                    results['avg_similarity'].append(np.mean(similarities))
                    results['top_similarity'].append(np.max(similarities))
                    results['similarity_std'].append(np.std(similarities))
                    
                except Exception as e:
                    print(f"Error for user {user_id}, track {track_idx}: {e}")
                    continue
        
        return pd.DataFrame(results)
    
    def analyze_ab_results(self, results_df):
        """Analyze live AB test results"""
        print("=== Live A/B Test Results Analysis ===\n")
        
        # Performance by variant
        variant_summary = results_df.groupby('variant').agg({
            'avg_similarity': ['mean', 'std', 'count'],
            'top_similarity': ['mean', 'std'],
            'similarity_std': ['mean', 'std']
        }).round(4)
        
        print("Performance by Variant:")
        print(variant_summary)
        print("\n")
        
        # Statistical significance test
        control_group = results_df[results_df['variant'] == 'control']['avg_similarity']
        treatment_group = results_df[results_df['variant'] == 'treatment']['avg_similarity']
        
        if len(control_group) > 0 and len(treatment_group) > 0:
            t_stat, p_value = stats.ttest_ind(control_group, treatment_group)
            effect_size = (treatment_group.mean() - control_group.mean()) / np.sqrt(
                ((len(treatment_group) - 1) * treatment_group.var() + 
                 (len(control_group) - 1) * control_group.var()) / 
                (len(treatment_group) + len(control_group) - 2)
            )
            
            print(f"Statistical Test Results:")
            print(f"Control mean: {control_group.mean():.4f}")
            print(f"Treatment mean: {treatment_group.mean():.4f}")
            print(f"Difference: {treatment_group.mean() - control_group.mean():.4f}")
            print(f"T-statistic: {t_stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Effect size (Cohen's d): {effect_size:.4f}")
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"Significance: {significance}")
        
        return variant_summary
    
    def cleanup(self):
        """Clean up AB testing resources"""
        if hasattr(self.ab_manager, 'close'):
            self.ab_manager.close()
    
    # Keep all your existing methods (prepare_features, two_stage_filtering, etc.)
    def prepare_features(self):
        """Prepare feature matrices for different recommendation approaches"""
        feature_cols = [col for col in self.df.columns if col != 'track_id']
        
        # Full feature set
        self.features_full = self.scaler.fit_transform(self.df[feature_cols])
        
        # High-reliability features (tempo, chroma, spectral_centroid)
        reliable_features = []
        for col in feature_cols:
            if 'tempo' in col.lower() or 'chroma' in col.lower() or 'spectral_centroid' in col.lower():
                reliable_features.append(col)
        
        if reliable_features:
            self.features_reliable = self.scaler.fit_transform(self.df[reliable_features])
        else:
            print("Using feature indices - adjust based on your CSV structure")
            self.features_reliable = self.features_full[:, -14:]
            
        self.tempo_chroma_features = self.features_reliable
        
    def two_stage_filtering(self, query_track_idx, tempo_tolerance=15, chroma_threshold=0.8):
        """Two-stage filtering approach"""
        query_features = self.features_reliable[query_track_idx]
        similarities = cosine_similarity([query_features], self.features_reliable)[0]
        candidate_indices = np.argsort(similarities)[::-1]
        recommendations = [idx for idx in candidate_indices if idx != query_track_idx][:10]
        return recommendations, similarities[recommendations]
    
    def standard_recommendation(self, query_track_idx):
        """Standard recommendation using all features"""
        query_features = self.features_full[query_track_idx]
        similarities = cosine_similarity([query_features], self.features_full)[0]
        candidate_indices = np.argsort(similarities)[::-1]
        recommendations = [idx for idx in candidate_indices if idx != query_track_idx][:10]
        return recommendations, similarities[recommendations]
    
    def reliable_features_recommendation(self, query_track_idx):
        """Recommendation using only high-reliability features"""
        query_features = self.features_reliable[query_track_idx]
        similarities = cosine_similarity([query_features], self.features_reliable)[0]
        candidate_indices = np.argsort(similarities)[::-1]
        recommendations = [idx for idx in candidate_indices if idx != query_track_idx][:10]
        return recommendations, similarities[recommendations]

# Example usage with AB testing:
"""
# Load your CSV data
df = pd.read_csv('your_music_features.csv')

# Initialize tester with AB testing
tester = MusicRecommendationABTester(df, user_id="user123")

# Create a custom experiment
custom_experiment = ExperimentConfig(
    name="advanced_recommendation_test",
    description="Test advanced filtering vs standard",
    status=ExperimentStatus.ACTIVE,
    start_date=datetime.now(),
    variants={"control": 0.4, "treatment_a": 0.3, "treatment_b": 0.3},
    primary_metric=MetricType.SIMILARITY_SCORE,
    secondary_metrics=[MetricType.CLICK_THROUGH_RATE, MetricType.USER_SATISFACTION],
    target_sample_size=500,
    minimum_detectable_effect=0.03
)
tester.ab_manager.create_experiment(custom_experiment)

# Get recommendations (automatically uses AB testing)
recommendations, similarities = tester.get_recommendations(query_track_idx=0)

# Track user interactions with proper metrics
tester.track_user_interaction("recommendation_algorithm", "click", track_idx=recommendations[0])
tester.track_user_interaction("recommendation_algorithm", "play", track_idx=recommendations[0], 
                            metadata={"play_duration": 120})

# Track custom metrics
tester.ab_tester.track_metric("recommendation_algorithm", MetricType.USER_SATISFACTION, 4.5)

# Run live AB test with multiple users
user_ids = ["user1", "user2", "user3", "user4", "user5"]
test_tracks = [0, 1, 2, 3, 4]
results = tester.run_live_ab_test(user_ids, test_tracks)

# Analyze results
summary = tester.analyze_ab_results(results)

# Check experiment status
experiment = tester.ab_manager.get_experiment("recommendation_algorithm")
print(f"Experiment active: {experiment.is_active()}")
print(f"Primary metric: {experiment.primary_metric}")

# Update experiment status
tester.ab_manager.update_experiment_status("recommendation_algorithm", ExperimentStatus.COMPLETED)

# Clean up
tester.cleanup()
"""