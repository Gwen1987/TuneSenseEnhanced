import gradio as gr
from downloader_core import search_ytmusic, download_audio, get_recommendations, get_radar_plot
from extract_features_final import extract_features

with gr.Blocks(title="Enhanced Youtube Downloader", theme=gr.themes.Soft()) as demo: #keep

    user_id = "some_uuid"  # Placeholder for user ID for AB Testing

    gr.Markdown("# 🎵 Enhanced YouTube Downloader")

    #User search bar
    with gr.Row():
        search_input = gr.Textbox(placeholder = "Search for a song or artist", label="Search YT Music", lines=1)
        search_button = gr.Button("Search")

    #Progress display
    progress_display = gr.Textbox(label="Status", interactive=False)

    #Youtube results
    with gr.Column():
        gr.Markdown("## Search Results")
        search_gallery = gr.Gallery(label="Search Results", columns=5, rows=1, height=200, allow_preview=False)
        selected_info = gr.Textbox(label="Selected Info", interactive=False)
                      

    #Recommendation block
    with gr.Column():
        gr.Markdown("### Recommended Songs")
        recommendation_output = gr.Textbox(label="Top 3 Recommendations", lines=3)

    #Radar chart display
    radar_plot = gr.Plot(label="🎯 Audio Feature Breakdown")

    #=== Backend logic ===

    search_results = []  # Global variable for now

    def handle_search(query):
        global search_results
        search_results = search_ytmusic(query)
        return [(item["thumbnail"], f"{item['title']} — {item['duration']}") for item in search_results]  # Thumbnails with captions

    def handle_gallery_select(evt: gr.SelectData):
            index = evt.index
            selected = search_results[index]
            if "video_id" not in selected:
                print(f"❌ (app.handle_gallery_select) Missing video_id in: {selected}")
                return "❌ (app.handle_gallery_select) Error: Missing video ID", "", None, None
            
            video_id = selected["video_id"]
            text = f"{selected['title']} — {selected['duration']}"
            print(f"🎯 (app.handle_gallery_select)Selected video_id: {video_id}")


            status, recommendations, radar = handle_download(video_id)
            return text, status, recommendations, radar
    
    def handle_download(video_id):
        print(f"🔻(app.handle_download) DEBUG: video_id = {video_id} (type: {type(video_id)})")

        status, filepath = download_audio(video_id)
       
        if not filepath:
            return status, "", None  # or any fallback values
        
        features = extract_features(filepath)        # <-- if you're using librosa here
        if not features:
            return "❌ (app.handle_download) Feature extraction failed", "", None

        recommendations = get_recommendations(features) #May need to update
        radar = get_radar_plot(features)
        return status, recommendations, radar
    
    



    
    
    # def handle_thumbnail_select(evt: gr.SelectData):
    #     index = evt.index
    #     video_id = search_results[index]["video_id"]
    #     return handle_download(video_id)
    
    
    #Bind callbacks

    search_button.click(fn=handle_search, inputs=search_input, outputs=search_gallery)

    # search_button.click(
    #     fn=handle_search,
    #     inputs=search_input,
    #     outputs=search_gallery
    # )

    search_gallery.select(
        fn=handle_gallery_select,
        inputs=[],
        outputs=[selected_info, progress_display, recommendation_output, radar_plot]

    )
    '''
    def handle_song_click(video_id):
        return f"Downloading song: {title}..."
    
    with gr.Blocks() as demo:
        status = gr.Textbox(label="Status")

        with gr.Row():
            for title, duration, thumbnail_url, video_id in search_results:
                with gr.Column():
                    btn = gr.Button(visible=False) #Hidden button to simulate click
                    img = gr.Image(value=thumbnail_url, width=100, height=100, show_label=False)
                    label = gr.Markdown(f"**{title}**,br.{duration}")
                    #Wrap both in a clickable label
                    img.select(fn=lambda v=video_id: handle_song_click(v), inputs=[], outputs=status)
                    label.select(fn=lambda v=video_id: handle_song_click(v), inputs=[], outputs=status)
'''
demo.launch(share=True)