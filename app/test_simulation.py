from downloader_core import (
    search_ytmusic,
    download_audio,
    get_audio_features,
    get_recommendations,
    assign_variant,
    log_ab_event
)
from uuid import uuid4
import time

def simulate_ab_test(song_queries, max_per_query=1):
    print("\n🎯 Starting AB test simulation...\n")

    for query in song_queries:
        print(f"\n🔍 Query: {query}")
        results = search_ytmusic(query)

        if not results:
            print("❌ No results found.")
            continue

        for song in results[:max_per_query]:
            video_id = song.get("video_id")
            title = song.get("title")
            artist = song.get("artist")

            print(f"🎵 {title} – {artist} | video_id={video_id}")

            status, filepath = download_audio(video_id)
            if not filepath:
                print(f"❌ Failed to download {title}")
                continue

            features = get_audio_features(video_id)
            if not features:
                print(f"❌ Failed to extract features for {video_id}")
                continue

            user_id = str(uuid4())
            variant = assign_variant(user_id)

            try:
                recs = get_recommendations(features, variant=variant)
                print(f"✅ Recommendations for variant {variant}:")
                for rec in recs:
                    print("  🔸", rec)

                # ✅ Log event to MongoDB Atlas
                log_ab_event(
                    user_id=user_id,
                    test_name="recommendation_knn_vs_cosine",
                    variant=variant,
                    event_type="recommendation_tested",
                    metadata={
                        "query": query,
                        "video_id": video_id,
                        "title": title,
                        "artist": artist,
                        "num_recs": len(recs)
                    }
                )

            except Exception as e:
                print(f"❌ Error during recommendation: {e}")

            time.sleep(2)

# === Sample usage
if __name__ == "__main__":
    test_queries = [
        "Taylor Swift",
        "Arctic Monkeys",
        "Adele",
        "Drake",
        "Daft Punk"
    ]
    simulate_ab_test(test_queries, max_per_query=2)
