import os
import signal
import traceback
from io import BytesIO

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from inference import Inference

APP = FastAPI()

tts_pipline = Inference()


class TTSRequest(BaseModel):
    text: str = None
    speaker_id: str = ""
    trace_id: str = ""


def pack_wav(data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    io_buffer.seek(0)
    return io_buffer


def pack_mp3(data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='mp3')
    io_buffer.seek(0)
    return io_buffer


async def tts_handle(trace_id, text, speaker_id):
    try:
        sr, audio = tts_pipline.inference(trace_id, text, speaker_id)
        audio_data = pack_mp3(audio, sr).getvalue()
        return Response(audio_data, media_type=f"audio/mp3")
    except Exception as ex:
        return JSONResponse(status_code=500, content=f"ERROR:{ex}")


async def tts_mp3_base64_handle(trace_id, text, speaker_id):
    try:
        sr, audio = tts_pipline.inference(trace_id, text, speaker_id)
        audio_data = pack_wav(BytesIO(), audio, sr).getvalue()
        return Response(audio_data, media_type=f"audio/wav")
    except Exception as ex:
        return JSONResponse(status_code=500, content=f"ERROR:{ex}")


@APP.get("/api/check-health")
async def check_health():
    return "OK"


@APP.get("/tts")
async def tts(trace_id: str, text: str, speaker_id: str):
    return await tts_handle(trace_id, text, speaker_id)


@APP.post("/audgeneratebase64")
async def tts(request: TTSRequest):
    return await tts_mp3_base64_handle(request, request.text, request.speaker_id)


if __name__ == "__main__":
    try:
        uvicorn.run(app=APP, host="0.0.0.0", port=7080, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
