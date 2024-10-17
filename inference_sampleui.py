# import os
#
# import gradio as gr
#
# from inference import Inference
#
# root_path = os.path.split(os.path.realpath(__file__))[0]
# print("__file__", __file__)
# print("root_path", root_path)
#
# tts_pipline = Inference()
#
#
# def tts_fn(text, speaker):
#     try:
#         sr, audio = tts_pipline.inference("", text, speaker)
#         return "Success", (sr, audio)
#     except Exception as e:
#         return f"Error: {e}", None
#
#
# if __name__ == "__main__":
#     speakers = tts_pipline.get_speakers()
#     app = gr.Blocks()
#     with app:
#         with gr.Tab("Text-to-Speech"):
#             with gr.Row():
#                 with gr.Column():
#                     textbox = gr.TextArea(label="Text",
#                                           placeholder="Type your sentence here",
#                                           value="Hey there! Feel free to share your thoughts or any interesting "
#                                                 "stories. I'm all ears!",
#                                           elem_id=f"tts-input")
#                     # select character
#                     char_dropdown = gr.Dropdown(choices=speakers, value=speakers[0], label='character')
#                 with gr.Column():
#                     text_output = gr.Textbox(label="Message")
#                     audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
#                     btn = gr.Button("Generate!")
#                     btn.click(tts_fn,
#                               inputs=[textbox, char_dropdown],
#                               outputs=[text_output, audio_output])
#     app.launch(share=False, server_port=7080, server_name="0.0.0.0")
