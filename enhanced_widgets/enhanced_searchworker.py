from PyQt6.QtCore import QThread, pyqtSignal
import yt_dlp

class SearchWorker(QThread):
    results_ready = pyqtSignal(list)

    def __init__(self, query):
        super().__init__()
        self.query = query

    def run(self):
        ydl_opts = {
            'quiet': False,
            'skip_download': True,
            'extract_flat': True,
            'default_search': 'ytsearch5',
            'nocheckcertificate': True,
            'force_generic_extractor': False,
            'simulate': True,
            'forcejson': True,
        }

        try:
            print(f"🔍 Searching: ytsearch5:{self.query}")
            ydl = yt_dlp.YoutubeDL(ydl_opts)
            info = ydl.extract_info(f"ytsearch5:{self.query}", download=False)

            raw_results = info.get("entries", [info]) if "entries" in info else [info]
            processed_results = []

            for video in raw_results:
                # Get thumbnail URL from first entry in thumbnails list
                thumbnail_url = video.get("thumbnail")
                if not thumbnail_url:
                    thumbnails = video.get("thumbnails", [])
                    if thumbnails:
                        thumbnail_url = thumbnails[0].get("url")

                video_info = {
                    "title": video.get("title"),
                    "url": video.get("webpage_url"),
                    "duration": video.get("duration_string", ""),
                    "channel": video.get("channel", ""),
                    "thumbnail": thumbnail_url or "",  # default to empty string
                }

                processed_results.append(video_info)

            self.results_ready.emit(processed_results)

        except Exception as e:
            print(f"❌ SearchWorker error: {e}")
            self.results_ready.emit([])
