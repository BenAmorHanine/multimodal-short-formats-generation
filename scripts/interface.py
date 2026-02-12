import gradio as gr
import os
import subprocess

def creative_ai_clipper(video_path, user_prompt, num_shorts, duration):
    if video_path is None or not user_prompt:
        return [None] * 5
    
    num_shorts = int(num_shorts)
    duration = int(duration)
    output_dir = "custom_shorts"
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    # Timestamps simulés pour le test
    dummy_timestamps = ["00:00:05", "00:00:25", "00:01:10", "00:01:50", "00:02:30"]

    for i in range(min(num_shorts, len(dummy_timestamps))):
        out_file = os.path.join(output_dir, f"short_{i+1}.mp4")
        start = dummy_timestamps[i]
        cmd = f"ffmpeg -y -ss {start} -t {duration} -i {video_path} -c:v libx264 -c:a aac -loglevel quiet {out_file}"
        subprocess.run(cmd, shell=True)
        results.append(out_file)
    
    return results + [None] * (5 - len(results))

def launch_interface():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", secondary_hue="gray")) as demo:
        gr.HTML('<div style="text-align: center; padding: 20px; background-color: #2d2d2d; border-radius: 10px; margin-bottom: 20px;"><h1 style="color: #ff9800; margin-bottom: 0;">SMART SHORT GENERATOR</h1><p style="color: white;">Propulse par VideoLLaMA 2</p></div>')

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Video Source")
                video_input = gr.Video(label=None)
                with gr.Group():
                    gr.Markdown("### Parametres")
                    duration_slider = gr.Slider(minimum=4, maximum=15, step=2, value=6, label="Duree (sec)")
                    count_slider = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Nombre de clips")

            with gr.Column(scale=2):
                gr.Markdown("### Resultats extraits")
                outputs = [gr.Video(label=f"Top {i+1}", interactive=False) for i in range(5)]
                with gr.Row():
                    outputs[0]
                    outputs[1]
                with gr.Row():
                    outputs[2]
                    outputs[3]
                    outputs[4]

        with gr.Row(variant="panel"):
            with gr.Column(scale=4):
                user_instruction = gr.Textbox(label="Que voulez-vous extraire ?", placeholder="Ex: Action, sourires...", lines=1)
            with gr.Column(scale=1):
                generate_btn = gr.Button("GENERER", variant="primary", size="lg")

        generate_btn.click(
            fn=creative_ai_clipper,
            inputs=[video_input, user_instruction, count_slider, duration_slider],
            outputs=outputs
        )

    demo.launch(share=True)