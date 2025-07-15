import gradio as gr
from downloader_core import search_ytmusic, download_audio, get_recommendations, get_radar_plot

with gr.Blocks(title="Enhanced Youtube Downloader", theme=gr.themes.Soft()) as demo:

    user_id = "some_uuid"  # Placeholder for user ID for AB Testing

    gr.Markdown("# 🎵 Enhanced YouTube Downloader")

    #User search bar
    with gr.Row():
        search_input = gr.Textbox(placeholder = "Search for a song or artist", label="Search YT Music", lines=1)
        search_button = gr.Button("Search")

    #Progress display
    progress_display = gr.Textbox(label="Status", interactive=False, lines=1)

    #Youtube results
    with gr.Column():
        gr.Markdown("## Search Results")
        yt_buttons = [gr.Button(f"Result {i+1}") for i in range(5)]

    #Recommendation block
    with gr.Column():
        gr.Markdown("### Recommended Songs")
        recommendation_output = gr.Textbox(label="Top 3 Recommendations", lines=3)

    #Radar chart display
    radar_plot = gr.Plot(label="🎯 Audio Feature Breakdown")
                      
    #=== Callback logic ===

    def handle_search(query):
        #returns 5 video titles/ID's
        results = search_ytmusic(query)
        return results #List of strings for the button labels
    
    def handle_download(video_title):
        progress_display.update("Downloading...")
        status = download_audio(video_title)
        recommendations = get_recommendations(video_title)
        radar = get_radar_plot(video_title)
        return status, recommendations, radar
    
    #Bind callbacks
    search_button.click(
        fn=handle_search,
        inputs=search_input,
        outputs=yt_buttons
    )

    for i, button in enumerate(yt_buttons):
        button.click(
            fn=lambda title:handle_download(title),
            inputs=[],
            outputs=[progress_display, recommendation_output, radar_plot]
        )

demo.launch()   