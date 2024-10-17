import os
import time

import gradio as gr

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

root_path = os.path.split(os.path.realpath(__file__))[0]
print("__file__", __file__)
print("root_path", root_path)

speakers = [
    "Binary",
    "Dara",
    "Fantasm",
    "MaXine",
    "Neon",
    "Pyro",
    "Vigor",
    "Vio",
    "Ziggy"
]

tts_models = {}

for speaker in speakers:
    speaker_id = speaker.lower()
    cfg_path = f"GPT_SoVITS/mycfg/tts_infer_{speaker_id}.yaml"
    config = TTS_Config(cfg_path)
    tts_pipeline = TTS(config)
    tts_models[speaker_id] = tts_pipeline


def tts_fn(text, char):
    req = {
        "text": text,
        "text_lang": "en",
        "text_split_method": "cut5",
        "media_type": "wav",
        "batch_size": 1,
        "ref_audio_path": f"/workspace/res/gptsovits-930/{char}/{char}.wav",
    }

    start = time.time()
    tts_generator = tts_models[char.lower()].run(req)
    sr, audio_data = next(tts_generator)
    end = time.time()

    print("INFERENCE_UI:", text, char, sr, end - start)
    return "Success", (sr, audio_data)


if __name__ == "__main__":
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
