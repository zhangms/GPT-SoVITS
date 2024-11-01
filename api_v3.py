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
from inference_sampleui import SampleUI

tts_pipline = Inference()
sample_ui = SampleUI(tts_pipline)

APP = FastAPI()
gr_app = sample_ui.gr_app()
gr.mount_gradio_app(APP, gr_app, path="/gr", )


class TTSRequest(BaseModel):
    text: str = None
    speaker_id: str = ""
    trace_id: str = ""
    fmt: str = "mp3"


def pack_voice(data: np.ndarray, rate: int, fmt: str):
    io_buffer = BytesIO()
    if fmt.lower() == "wav":
        sf.write(io_buffer, data, rate, format='wav')
    elif fmt.lower() == "mp3":
        sf.write(io_buffer, data, rate, format='mp3')
    else:
        sf.write(io_buffer, data, rate, format='wav')

    io_buffer.seek(0)
    return io_buffer


def pack_voice_b64(data: np.ndarray, rate: int, fmt: str):
    voice_data = pack_voice(data, rate, fmt).getvalue()
    encoded_content = base64.b64encode(voice_data)
    return b'data:' + encoded_content + b'\n'


async def tts_handle(trace_id, speaker_id, text, fmt):
    try:
        sr, audio = tts_pipline.inference(trace_id, speaker_id, text)
        audio_data = pack_voice(audio, sr, fmt).getvalue()
        return Response(audio_data, media_type=f"audio/{fmt}")
    except Exception as ex:
        return JSONResponse(status_code=500, content=f"TTS_SERVICE_ERROR:{ex}")


async def tts_stream_handle(trace_id, speaker_id, text, fmt):
    try:
        tts_gen = tts_pipline.generator(speaker_id, text)

        def streaming_generator(tts_generator):
            start = time.time()
            index = 0
            for sr, chunk in tts_generator:
                index += 1
                end = time.time()
                rt = end - start
                print(f"{datetime.datetime.now()}|TTS_STREAM_GENERATOR|{trace_id}|{speaker_id}|index:{index}|rt:{rt}")
                yield pack_voice_b64(chunk, sr, fmt)
            print("", flush=True)

        return StreamingResponse(streaming_generator(tts_gen), media_type="text/event-stream")
    except Exception as ex:
        return JSONResponse(status_code=500, content=f"TTS_SERVICE_ERROR:{ex}")


@APP.get("/api/check-health")
async def check_health():
    return "OK"


@APP.get("/tts")
async def tts(trace_id: str, speaker_id: str, text: str, fmt: str):
    return await tts_handle(trace_id, speaker_id, text, fmt)


@APP.post("/tts-post")
async def tts_post(request: TTSRequest):
    return await tts_handle(request.trace_id, request.speaker_id, request.text, request.fmt)


@APP.get("/tts_stream")
async def tts_stream(trace_id: str, speaker_id: str, text: str, fmt: str):
    print(f"{datetime.datetime.now()}|TTS_STREAM|{trace_id}|{speaker_id}|{fmt}|{text}")
    return await tts_stream_handle(trace_id, speaker_id, text, fmt)


if __name__ == "__main__":
    try:
        uvicorn.run(APP, host="0.0.0.0", port=7080, workers=1)
    except Exception as e:
        print(e)
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
