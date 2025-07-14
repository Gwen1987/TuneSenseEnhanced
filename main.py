from PyQt6.QtWidgets import QApplication
import sys
from enhanced_widgets.enhanced_downloader_ui import EnhancedYouTubeDownloader

if __name__ == "__main__":
    app = QApplication(sys.argv)  # MUST come first
    window = EnhancedYouTubeDownloader()  # Only now can you create QWidget
    window.show()
    sys.exit(app.exec())
