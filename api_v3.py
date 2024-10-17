import base64
import datetime
import os
import signal
import time
import traceback
from io import BytesIO

import gradio as gr
import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from inference import Inference

APP = FastAPI()

tts_pipline = Inference()


class TTSRequest(BaseModel):
    text: str = None
    speaker_id: str = ""
    trace_id: str = ""


def pack_mp3(data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='mp3')
    io_buffer.seek(0)
    return io_buffer


def tts_fn(text, speaker):
    try:
        sr, audio = tts_pipline.inference("", text, speaker)
        return "Success", (sr, audio)
    except Exception as ex:
        return f"Error: {ex}", None


async def tts_handle(trace_id, text, speaker_id):
    try:
        sr, audio = tts_pipline.inference(trace_id, text, speaker_id)
        audio_data = pack_mp3(audio, sr).getvalue()
        return Response(audio_data, media_type=f"audio/mp3")
    except Exception as ex:
        return JSONResponse(status_code=500, content=f"TTS_SERVICE_ERROR:{ex}")


async def tts_base64_handle(trace_id, text, speaker_id):
    try:
        sr, audio = tts_pipline.inference(trace_id, text, speaker_id)
        audio_data = pack_mp3(audio, sr).getvalue()
        encoded_content = base64.b64encode(audio_data)
        return {"audio": encoded_content}
    except Exception as ex:
        return JSONResponse(status_code=500, content=f"TTS_SERVICE_ERROR:{ex}")


async def tts_stream_handle(trace_id, text, speaker_id):
    try:
        tts_gen = tts_pipline.generator(text, speaker_id)
        print(f"{datetime.datetime.now()}|TTS_STREAM_GENERATOR|{trace_id}|{speaker_id}|{text}")

        def streaming_generator(tts_generator):
            start = time.time()
            index = 0
            for sr, chunk in tts_generator:
                end = time.time()
                rt = end - start
                start = end
                print(f"{datetime.datetime.now()}|TTS_STREAM_GENERATOR|{trace_id}|{speaker_id}|index:{index}|rt:{rt}")
                index += 1
                yield pack_mp3(chunk, sr).getvalue()

        return StreamingResponse(streaming_generator(tts_gen), media_type=f"audio/mp3")
    except Exception as ex:
        return JSONResponse(status_code=500, content=f"TTS_SERVICE_ERROR:{ex}")


@APP.get("/api/check-health")
async def check_health():
    return "OK"


@APP.get("/tts")
async def tts(trace_id: str, text: str, speaker_id: str):
    return await tts_handle(trace_id, text, speaker_id)


@APP.post("/audgeneratebase64")
async def audgeneratebase64(request: TTSRequest):
    return await tts_base64_handle(request.trace_id, request.text, request.speaker_id)


@APP.get("/tts_stream")
async def tts_stream(trace_id: str, text: str, speaker_id: str):
    return await tts_stream_handle(trace_id, text, speaker_id)


def gr_app():
    speakers = tts_pipline.get_speakers()
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
                    btn.click(tts_fn,
                              inputs=[textbox, char_dropdown],
                              outputs=[text_output, audio_output])
    return app


if __name__ == "__main__":
    try:
        io = gr_app()
        gr.mount_gradio_app(APP, io, path="/gr")
        uvicorn.run(app=APP, host="0.0.0.0", port=7080, workers=1)
    except Exception as e:
        print(e)
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
