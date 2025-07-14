import datetime
from PyQt6.QtCore import Qt, QSize, QThreadPool, QTimer
from PyQt6.QtGui import QPixmap, QIcon, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem, QMessageBox, QLabel,
    QHBoxLayout, QCheckBox, QTextEdit, QScrollArea
)
from enhanced_widgets.enhanced_searchworker import SearchWorker
from enhanced_widgets.enhanced_thumbnailthread import (
    get_audio_features,
    fetch_metadata,
    RadarChartCanvas, ThumbnailSignalEmitter, ThumbnailWorker
)
from enhanced_widgets.enhanced_recommendationworker import EnhancedRecommendationWorker
from ab_testing_framework import ABTestManager, TuneSenseABTester
import uuid
import os
import json
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"


class EnhancedYouTubeDownloader(QWidget):
    """Enhanced YouTube downloader with AB testing capabilities"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TuneSense 🎵🎶 (AB Testing Enhanced)")
        self.setGeometry(300, 300, 900, 700)
        
        # Initialize AB testing
        self.user_id = f"user_{uuid.uuid4().hex[:8]}"
        mongo_uri = "mongodb+srv://supertrooper:UofT1234@musiccluster.ix1va8y.mongodb.net/?retryWrites=true&w=majority&appName=musiccluster"
        self.ab_manager = ABTestManager(mongo_uri)
        self.ab_tester = TuneSenseABTester(self.ab_manager)
        self.ab_tester.set_user(self.user_id)
        
        self.setup_ui()
        self.setup_ab_testing()
        
        self.threadpool = QThreadPool()
        self.thumbnail_emitter = ThumbnailSignalEmitter()
        self.thumbnail_emitter.signal.connect(self.set_thumbnail)
        self.videos = []
        self.current_recommendations = []
        
        # Timer for updating AB test status
        self.ab_status_timer = QTimer()
        self.ab_status_timer.timeout.connect(self.update_ab_status)
        self.ab_status_timer.start(5000)  # Update every 5 seconds
    
    def setup_ui(self):
        """Set up the user interface"""
        self.layout = QVBoxLayout()
        
        # User ID display (for AB testing transparency)
        user_label = QLabel(f"User ID: {self.user_id}")
        user_label.setStyleSheet("color: gray; font-size: 10px;")
        self.layout.addWidget(user_label)
        
        # Search components
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search YouTube...")
        self.search_input.returnPressed.connect(self.start_search)
        self.layout.addWidget(self.search_input)
        
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.start_search)
        self.layout.addWidget(self.search_button)
        
        # Results list
        self.result_list = QListWidget()
        self.result_list.setIconSize(QSize(160, 90))
        self.result_list.itemClicked.connect(self.show_recommendations_and_chart)
        self.layout.addWidget(self.result_list)
        
        # Quality input
        self.quality_input = QLineEdit()
        self.quality_input.setPlaceholderText("Preferred MP3 quality (e.g., 192)")
        self.layout.addWidget(self.quality_input)
        
        # AB Testing info panel
        self.ab_info_label = QLabel("AB Test Status: Loading...")
        self.ab_info_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        self.layout.addWidget(self.ab_info_label)
        
        # Recommendations
        self.recommend_label = QLabel("🎧 Recommendations:")
        self.layout.addWidget(self.recommend_label)
        self.recommend_list = QListWidget()
        self.recommend_list.itemClicked.connect(self.on_recommendation_clicked)
        self.layout.addWidget(self.recommend_list)
        
        # Chart
        self.chart_placeholder = QLabel("📊 Radar Chart:")
        self.layout.addWidget(self.chart_placeholder)
        
        # AB Testing controls
        ab_controls = QHBoxLayout()
        self.show_variant_info = QCheckBox("Show AB Test Variant Info")
        self.show_variant_info.setChecked(True)
        self.show_variant_info.stateChanged.connect(self.toggle_variant_info)
        ab_controls.addWidget(self.show_variant_info)
        
        self.export_results_button = QPushButton("Export AB Test Results")
        self.export_results_button.clicked.connect(self.export_ab_results)
        ab_controls.addWidget(self.export_results_button)
        
        self.layout.addLayout(ab_controls)
        
        self.setLayout(self.layout)
    
    def setup_ab_testing(self):
        """Initialize AB testing experiment if not exists"""
        try:
            from ab_testing_framework import ExperimentConfig, MetricType, ExperimentStatus
            from datetime import datetime
            
            experiment_config = ExperimentConfig(
                name="recommendation_algorithm_v1",
                description="Test weighted KNN (k=5) vs unweighted KNN (k=3)",
                status=ExperimentStatus.ACTIVE,
                start_date=datetime.utcnow(),
                variants={"control": 0.5, "treatment": 0.5},
                primary_metric=MetricType.CLICK_THROUGH_RATE,
                secondary_metrics=[MetricType.RECOMMENDATION_DIVERSITY],
                created_by="TuneSense",
)
            
            # Try to create experiment (will fail if already exists, which is fine)
            try:
                self.ab_manager.create_experiment(experiment_config)
                print("✅ AB Test experiment created successfully")
            except Exception as e:
                print(f"ℹ️  AB Test experiment already exists or creation failed: {e}")
            
            # Update initial AB status
            self.update_ab_status()
            
        except Exception as e:
            print(f"❌ AB Testing setup error: {e}")
            self.ab_info_label.setText(f"AB Test Status: Error - {str(e)}")
    
    def update_ab_status(self):
        """Update AB testing status display"""
        try:
            variant = self.ab_tester.get_variant("recommendation_algorithm_v1")
            config = self.ab_tester.get_recommendation_config("recommendation_algorithm_v1")

            status_text = f"AB Test Status: Active | Variant: {variant} | Config: k={config['k']}, weighted={config['weighted']}"
            self.ab_info_label.setText(status_text)

            
        except Exception as e:
            self.ab_info_label.setText(f"AB Test Status: Error - {str(e)}")
    
    def start_search(self):
        """Start search with AB testing event logging"""
        query = self.search_input.text().strip()
        if not query:
            return
        
        # Log search event for AB testing
        self.ab_tester.log_search_query(query)
        
        self.search_button.setEnabled(False)
        self.search_button.setText("Searching...")
        self.result_list.clear()
        
        # Start search worker
        self.search_worker = SearchWorker(query)
        self.search_worker.results_ready.connect(self.display_results)
        self.search_worker.finished.connect(self.search_finished)
        self.search_worker.start()
    
    def display_results(self, videos):
        """Display search results"""
        self.videos = videos
        self.result_list.clear()
        
        for video in videos:
            item = QListWidgetItem(f"{video['title']} - {video['channel']}")
            item.setData(Qt.ItemDataRole.UserRole, video)
            self.result_list.addItem(item)
            
            # Load thumbnail asynchronously
        thumbnail_worker = ThumbnailWorker(
            video['thumbnail'],
            self.thumbnail_emitter,
            video.get('_ab_variant')  # ← this safely adds the variant if present
)            #self.threadpool.start(thumbnail_worker)
    
    def search_finished(self):
        """Handle search completion"""
        self.search_button.setEnabled(True)
        self.search_button.setText("Search")
    
    def set_thumbnail(self, url, pixmap):
        """Set thumbnail for video item"""
        for i in range(self.result_list.count()):
            item = self.result_list.item(i)
            video = item.data(Qt.ItemDataRole.UserRole)
            if video and video['thumbnail'] == url:
                item.setIcon(QIcon(pixmap))
                break
    
    def show_recommendations_and_chart(self, item):
        """Show recommendations and chart for selected video"""
        video = item.data(Qt.ItemDataRole.UserRole)
        if not video:
            return

        # Log video selection for AB testing
        self.ab_tester.log_video_selection(video)

        self.recommend_list.clear()

        try:
            # Extract video ID and title
            video_id = video.get("id") or video.get("url")
            title = video.get("title")

            if not title:
                QMessageBox.warning(self, "Warning", "Video title missing. Cannot extract features.")
                return

            # Get audio features
            audio_features = get_audio_features(video['url'], video['title'])

            if audio_features is None:
                QMessageBox.warning(self, "Warning", "Could not extract audio features")
                return

            # Initialize or import vectorizer before using it
            from enhanced_widgets.enhanced_recommendationworker import vectorizer
            vectors = vectorizer.transform([audio_features])  # ✅ now safe

            # Get metadata
            metadata = fetch_metadata(video_id)
            if metadata:
                audio_features.update(metadata)

            # Start recommendation worker
            self.recommendation_worker = EnhancedRecommendationWorker(
                audio_features,
                self.user_id
            )
            self.recommendation_worker.recommendations_ready.connect(self.display_recommendations)
            self.recommendation_worker.start()

            # Create and display radar chart
            self.create_radar_chart(audio_features)

        except Exception as e:
            print(f"❌ Error processing video: {e}")
            QMessageBox.critical(self, "Error", f"Failed to process video: {str(e)}")

    def display_recommendations(self, recommendations):
        """Display recommendations with AB testing info"""
        self.current_recommendations = recommendations
        self.recommend_list.clear()
        
        for i, rec in enumerate(recommendations):
            display_text = f"{rec.get('track_name', 'Unknown')} by {rec.get('artist_name', 'Unknown')}"
            
            # Add AB testing info if enabled
            if self.show_variant_info.isChecked() and '_ab_variant' in rec:
                variant_info = rec['_ab_variant']
                display_text += f" [Variant: k={variant_info['k']}, weighted={variant_info['weighted']}]"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, rec)
            self.recommend_list.addItem(item)
    
    def on_recommendation_clicked(self, item):
        """Handle recommendation click with AB testing logging"""
        recommendation = item.data(Qt.ItemDataRole.UserRole)
        if not recommendation:
            return
        
        # Log recommendation click for AB testing
        self.ab_tester.log_recommendation_click(recommendation)
        
        # Show recommendation details
        track_name = recommendation.get('track_name', 'Unknown')
        artist_name = recommendation.get('artist_name', 'Unknown')
        
        msg = QMessageBox()
        msg.setWindowTitle("Recommendation Details")
        msg.setText(f"Track: {track_name}\nArtist: {artist_name}")
        
        if self.show_variant_info.isChecked() and '_ab_variant' in recommendation:
            variant_info = recommendation['_ab_variant']
            msg.setInformativeText(f"AB Test Variant: k={variant_info['k']}, weighted={variant_info['weighted']}")
        
        msg.exec()
    
    def create_radar_chart(self, audio_features):
        """Create radar chart for audio features"""
        try:
            # Remove existing chart if present
            for i in range(self.layout.count()):
                widget = self.layout.itemAt(i).widget()
                if isinstance(widget, RadarChartCanvas):
                    widget.deleteLater()
                    break
            
            # Create new radar chart
            chart = RadarChartCanvas(audio_features)
            chart.setMaximumHeight(300)
            
            # Insert chart after the chart placeholder
            placeholder_index = 0
            for i in range(self.layout.count()):
                widget = self.layout.itemAt(i).widget()
                if widget == self.chart_placeholder:
                    placeholder_index = i + 1
                    break
            
            self.layout.insertWidget(placeholder_index, chart)
            
        except Exception as e:
            print(f"❌ Error creating radar chart: {e}")
    
    def toggle_variant_info(self, state):
        """Toggle AB testing variant info display"""
        # Refresh recommendations display
        if self.current_recommendations:
            self.display_recommendations(self.current_recommendations)
    
    def export_ab_results(self):
        """Export AB testing results"""
        try:
            # Get AB test results
            results = self.ab_manager.get_experiment_results("recommendation_algorithm_v1")
            
            # Create export dialog
            dialog = QMessageBox()
            dialog.setWindowTitle("AB Test Results Export")
            
            # Format results for display
            if results:
                results_text = json.dumps(results, indent=2, default=str)
                
                # Create scrollable text area
                text_widget = QTextEdit()
                text_widget.setPlainText(results_text)
                text_widget.setReadOnly(True)
                text_widget.setMaximumHeight(400)
                
                # Save to file
                filename = f"ab_test_results_{self.user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                try:
                    with open(filename, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                    
                    dialog.setText(f"Results exported to: {filename}")
                    dialog.setDetailedText(results_text)
                    
                except Exception as e:
                    dialog.setText(f"Export failed: {str(e)}")
                    dialog.setDetailedText(results_text)
            else:
                dialog.setText("No AB test results available")
            
            dialog.exec()
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export AB test results: {str(e)}")
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Log session end for AB testing
        try:
            self.ab_tester.log_session_end()
        except Exception as e:
            print(f"❌ Error logging session end: {e}")
        
        # Stop timers
        if hasattr(self, 'ab_status_timer'):
            self.ab_status_timer.stop()
        
        event.accept()


# Additional utility functions for AB testing integration

def create_ab_test_report(ab_manager, experiment_id):
    """Create a comprehensive AB test report"""
    try:
        results = ab_manager.get_experiment_results(experiment_id)
        
        if not results:
            return "No results available for the experiment."
        
        report = f"""
AB Test Report: {experiment_id}
{'='*50}

Experiment Overview:
- Status: {results.get('status', 'Unknown')}
- Start Date: {results.get('start_date', 'Unknown')}
- Total Users: {results.get('total_users', 0)}

Variant Performance:
"""
        
        for variant, data in results.get('variants', {}).items():
            report += f"""
Variant {variant}:
- Users: {data.get('users', 0)}
- Conversion Rate: {data.get('conversion_rate', 0):.2%}
- Click Through Rate: {data.get('click_through_rate', 0):.2%}
- Recommendation Acceptance: {data.get('recommendation_acceptance', 0):.2%}
"""
        
        # Statistical significance
        if results.get('statistical_significance'):
            report += f"\n📊 Statistical Significance: {results['statistical_significance']}"
        
        return report
        
    except Exception as e:
        return f"Error generating report: {str(e)}"


def setup_ab_testing_dashboard(ab_manager):
    """Set up a simple AB testing dashboard"""
    try:
        experiments = ab_manager.get_all_experiments()
        
        dashboard = """
AB Testing Dashboard
==================

Active Experiments:
"""
        
        for exp in experiments:
            if exp.get('status') == 'ACTIVE':
                dashboard += f"- {exp['name']}: {exp['description']}\n"
        
        return dashboard
        
    except Exception as e:
        return f"Dashboard error: {str(e)}"



# Example usage and testing
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    
    # Create enhanced downloader with AB testing
    downloader = EnhancedYouTubeDownloader()
    downloader.show()
    
    sys.exit(app.exec())
