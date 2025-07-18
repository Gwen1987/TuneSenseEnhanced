import gradio as gr
from uuid import uuid4
from downloader_core import search_ytmusic, download_audio, get_recommendations, store_metadata
from extract_base import extract_features


with gr.Blocks(title="Enhanced Youtube Downloader", theme=gr.themes.Soft()) as demo:

    user_id = str(uuid4())

    gr.Markdown("# 🎵 Enhanced YouTube Downloader")

    # User search bar
    with gr.Row():
        search_input = gr.Textbox(
            placeholder="Search for a song or artist",
            label="Search YT Music",
            lines=1
        )
        search_button = gr.Button("Search")

    # Progress display
    progress_display = gr.Textbox(label="Status", interactive=False)

    # Gallery + info
    with gr.Column():
        gr.Markdown("## Search Results")

        search_gallery = gr.Gallery(
            label="Search Results",
            columns=5,
            rows=2,
            object_fit="contain",
            height=350,
            allow_preview=False,
            show_label=False
        )

        selected_info = gr.Textbox(label="Selected Info", interactive=False)

    # Recommendations
    with gr.Column():
        gr.Markdown("### Recommended Songs")
        recommendation_output = gr.Textbox(label="Top 3 Recommendations", lines=3)


    # === Backend logic ===

    search_results = []  # Global cache

    def handle_search(query):
        global search_results
        search_results = search_ytmusic(query)

        # Store metadata in Mongo
        for item in search_results:
            title = item.get("title", "")
            artist = item.get("artist", "")
            duration = item.get("duration", "")
            video_id = item.get("video_id", "")
            thumbnail_url = item.get("thumbnail", "")
            store_metadata({
                "video_id": video_id,
                "title": title,
                "artist": artist,
                "duration": duration,
                "thumbnail": thumbnail_url
            })

        return [(item["thumbnail"], f"{item['title']} — {item['duration']}") for item in search_results]

    def handle_gallery_select(evt: gr.SelectData):
        index = evt.index
        selected = search_results[index]

        if "video_id" not in selected:
            print(f"❌ (app.handle_gallery_select) Missing video_id in: {selected}")
            return "❌ Error: Missing video ID", "", []

        video_id = selected["video_id"]
        text = f"{selected['title']} — {selected['duration']}"
        print(f"🎯 Selected video_id: {video_id}")

        status, recommendations = handle_download(video_id)
        return text, status, recommendations

    def handle_download(video_id):
        print(f"🔻 Downloading... video_id = {video_id} (type: {type(video_id)})")

        status, filepath = download_audio(video_id)

        if not filepath:
            return status, "", None

        # Feature extraction
        base_features = extract_features(filepath)
        if not base_features:
            return "❌ Base feature extraction failed", "", None

        from extract_extra import extract_extra_features
        extra_features = extract_extra_features(filepath)
        if not extra_features:
            return "❌ Extra feature extraction failed", "", None

        features = {**base_features, **extra_features}

        recommendations = get_recommendations(features)
        return status, recommendations


    # === Bind events ===

    search_button.click(fn=handle_search, inputs=search_input, outputs=search_gallery)

    search_gallery.select(
        fn=handle_gallery_select,
        inputs=[],
        outputs=[selected_info, progress_display, recommendation_output]
    )

demo.launch()

