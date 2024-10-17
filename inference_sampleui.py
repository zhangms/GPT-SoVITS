import os

import gradio as gr

from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config, TTS

root_path = os.path.split(os.path.realpath(__file__))[0]
print("__file__", __file__)
print("root_path", root_path)

config = TTS_Config("GPT_SoVITS/mycfg/tts_infer_binary.yaml")
pipline = TTS(config)


def tts_fn(text, speaker):
    print("INFERENCE_UI:", text, speaker)
    return "Success", None


if __name__ == "__main__":
    # print("speakers:", predictor.get_speakers())
    # speakers = list(predictor.get_speakers().keys())
    speakers = ["Jack", "Rose"]
    app = gr.Blocks()
    with app:
        with gr.Tab("Text-to-Speech"):
            with gr.Row():
                with gr.Column():
                    textbox = gr.TextArea(label="Text",
                                          placeholder="Type your sentence here",
                                          value="Hey, how's it going? What's on your mind today?",
                                          elem_id=f"tts-input")
                    # select character
                    char_dropdown = gr.Dropdown(choices=speakers, value=speakers[0], label='character')
                with gr.Column():
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn = gr.Button("Generate!")
                    btn.click(tts_fn,
                              inputs=[textbox, char_dropdown],
                              outputs=[text_output, audio_output])
    app.launch(share=False, server_port=7080, server_name="0.0.0.0")
