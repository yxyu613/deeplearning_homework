import gradio as gr
from pathlib import Path
from PIL import Image
from run import extract_and_display_ske_cnt

with gr.Blocks() as iface:
    gr.Markdown("### Stroke2Sketch")
    gr.Markdown("Harnessing Stroke Attributes for Training-Free Sketch Generation.")
    with gr.Row():
        with gr.Column(scale=1):


            content_input = gr.Image(type='pil', label="Content Input")
            ref_input = gr.Image(type='pil', label="Reference Sketch Input")
            skip_steps = gr.Slider(1, 100, step=1, value=30, label='Skip Steps')
            gamma = gr.Slider(0, 1.0, step=0.01, value=0.25, label='Gamma')
            edge_method = gr.Dropdown(['teed', 'canny'], value='teed', label='Edge Method')
            edge_alpha = gr.Slider(0.1, 1.0, step=0.01, value=0.55, label='Edge Alpha')
            swap_guidance_scale = gr.Slider(0, 50.0, step=1, value=3.5, label='Swap Guidance Scale')
            contrast_strength = gr.Slider(0.5, 3.0, step=0.1, value=1.67, label='Contrast Strength')
            load_latents = gr.Checkbox(value=True, label='Load Latents')
            apply_mask = gr.Checkbox(value=True, label='Apply Mask')
            run_btn = gr.Button("Run")

        with gr.Column(scale=1):
            sketch_output = gr.Image(type='pil', label="Sketch Output")
            gr.Markdown("#### Example Candidates")
            gr.Examples(
                examples=[
                    [Image.open('data/cnt/cnt1.jpg'), Image.open('data/ref/ref1.jpg')],
                    [Image.open('data/cnt/cnt2.png'), Image.open('data/ref/ref2.jpg')],
                    [Image.open('data/cnt/cnt3.jpg'), Image.open('data/ref/ref3.jpg')],
                    [Image.open('data/cnt/cnt4.jpg'), Image.open('data/ref/ref4.jpg')],
                    [Image.open('data/cnt/cnt5.jpg'), Image.open('data/ref/ref5.jpg')],
                    [Image.open('data/cnt/cnt6.png'), Image.open('data/ref/ref6.jpg')],
                ],
                inputs=[content_input, ref_input],
                outputs=[sketch_output],
                label="Candidates",
            )

    run_btn.click(
        fn=extract_and_display_ske_cnt,
        inputs=[
            content_input, ref_input, skip_steps, gamma, edge_method, edge_alpha,
            swap_guidance_scale, contrast_strength, load_latents, apply_mask
        ],
        outputs=sketch_output
    )

if __name__ == "__main__":
    iface.launch(share=False)