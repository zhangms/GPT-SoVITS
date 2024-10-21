import gradio as gr

from inference import Inference


class SampleUI(object):
    def __init__(self, tts_pipline: Inference):
        self.tts_pipline = tts_pipline

    def tts_fn(self, text, speaker):
        try:
            sr, audio = self.tts_pipline.inference("", speaker, text)
            return "Success", (sr, audio)
        except Exception as ex:
            return f"Error: {ex}", None

    def gr_app(self):
        speakers = self.tts_pipline.get_speakers()
        app = gr.Blocks()
        with app:
            with gr.Tab("Text-to-Speech"):
                with gr.Row():
                    with gr.Column():
                        textbox = gr.TextArea(label="Text",
                                              placeholder="Type your sentence here",
                                              value="Hey there! Feel free to share your thoughts or any interesting "
                                                    "stories. I'm all ears!",
                                              elem_id=f"tts-input")
                        # select character
                        char_dropdown = gr.Dropdown(choices=speakers, value=speakers[0], label='character')
                    with gr.Column():
                        text_output = gr.Textbox(label="Message")
                        audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                        btn = gr.Button("Generate!")
                        btn.click(self.tts_fn,
                                  inputs=[textbox, char_dropdown],
                                  outputs=[text_output, audio_output])
        return app
